# code/model_pretrain.py
"""
MAE backbone and multimodal pretraining model (used by EFRM).

Implements:
- 2D patch embedding suitable for (channels x time) inputs,
- Masked AutoEncoder transformer (MAE) for EEG and fNIRS,
- Wrapper MAE class computing reconstruction + CLIP-like contrastive loss.
"""
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F
from thop import profile

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class PatchEmbed(nn.Module):
    """Patch embedding using Conv2d with kernel == patch size (for channel x time input)."""
    def __init__(self, img_size=(24,1024), patch_size=(1,32),
                 in_chans=1, embed_dim=768, norm_layer=None, flatten=True, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x

class MaskedAutoencoderViT(nn.Module):
    """MAE implementation using timm's Block transformer layers."""
    def __init__(self, img_size=(24,1024), patch_size=(1,32), in_chans=1,
                 mask_ratio=0.5, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.img_size = img_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size[0], self.grid_size[1], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size[0], self.grid_size[1], cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p_h = self.patch_embed.patch_size[0]
        p_w = self.patch_embed.patch_size[1]
        assert imgs.shape[2] % p_h == 0
        assert imgs.shape[3] % p_w == 0
        h = imgs.shape[2] // p_h
        w = imgs.shape[3] // p_w
        x = imgs.reshape((imgs.shape[0], imgs.shape[1], h, p_h, w, p_w))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape((imgs.shape[0], h * w, p_h * p_w * imgs.shape[1]))
        return x

    def unpatchify(self, x):
        p_h = self.patch_embed.patch_size[0]
        p_w = self.patch_embed.patch_size[1]
        h = self.img_size[0] // p_h
        w = self.img_size[1] // p_w
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p_h, p_w, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape((x.shape[0], x.shape[1], h * p_h, w * p_w))
        return imgs

    def random_masking(self, x):
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_embed(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :].mean(dim=1)
        return x

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss

class MAE(nn.Module):
    """
    Multimodal MAE wrapper used for pretraining:
      - eeg_model: MAE for EEG inputs (in_chans=1)
      - fnirs_model: MAE for fNIRS inputs (in_chans=2)
    Also provides a CLIP-like contrastive loss between pooled embeddings.
    """
    def __init__(self, eeg_size=(24,1024), fnirs_size=(64,128), mask_ratio=0.5):
        super().__init__()
        self.eeg_model = MaskedAutoencoderViT(img_size=eeg_size, mask_ratio=mask_ratio, in_chans=1)
        self.fnirs_model = MaskedAutoencoderViT(img_size=fnirs_size, mask_ratio=mask_ratio, in_chans=2)

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_clip_loss(self, feature1, feature2, logit_scale=0.1):
        feature1 = F.normalize(feature1, p=2, dim=-1)
        feature2 = F.normalize(feature2, p=2, dim=-1)
        logits1 = logit_scale * feature1 @ feature2.T
        logits2 = logit_scale * feature2 @ feature1.T
        logits1 = torch.clamp(logits1, min=-10, max=10)
        logits2 = torch.clamp(logits2, min=-10, max=10)
        labels = torch.arange(logits1.shape[0], device=feature1.device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits1, labels) +
            F.cross_entropy(logits2, labels)
        ) / 2
        return total_loss

    def forward(self, unpair_eeg, unpair_fnirs, pair_eeg, pair_fnirs):
        eeg_recon_loss = self.eeg_model(unpair_eeg)
        fnirs_recon_loss = self.fnirs_model(unpair_fnirs)
        eeg_h = self.eeg_model.forward_embed(pair_eeg)
        fnirs_h = self.fnirs_model.forward_embed(pair_fnirs)
        clip_loss = self.get_clip_loss(eeg_h, fnirs_h)
        return eeg_recon_loss, fnirs_recon_loss, clip_loss

def mae_vit_base(eeg_size=(24,1024), fnirs_size=(64,128), mask_ratio=0.5):
    model = MAE(eeg_size=eeg_size, fnirs_size=fnirs_size, mask_ratio=mask_ratio)
    return model