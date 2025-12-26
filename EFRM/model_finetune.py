# code/model_finetune.py
"""
Model definitions for fine-tuning / downstream tasks (EFRM).

This module reuses MAE implementation from model_pretrain and defines lightweight
classification wrappers that produce logits from encoder features.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pretrain import MAE, mae_vit_base

class EMA:
    """Simple EMA helper (kept from upstream)."""
    def __init__(self, beta, step_start_ema=2000):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.step_start_ema = step_start_ema

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        if self.step < self.step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class EEG_VisionTransformer(MAE):
    """
    Classification wrapper for EEG-only inputs.
    Produces logits via global average pooling over patch tokens.
    """
    def __init__(self, n_class, embed_dim=768, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(EEG_VisionTransformer, self).__init__()

        self.pos_drop = nn.Dropout(p=0.5)
        self.fc_norm = norm_layer(embed_dim)
        self.fc_class = nn.Linear(embed_dim, n_class)

    def forward_feature(self, x):
        """Extract features from EEG encoder and global-pool."""
        B = x.shape[0]
        x = self.eeg_model.patch_embed(x)

        cls_tokens = self.eeg_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.eeg_model.pos_embed
        x = self.pos_drop(x)

        for blk in self.eeg_model.blocks:
            x = blk(x)

        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        return x

    def forward_features(self, x):
        h = self.forward_feature(x)
        h = self.fc_norm(h)
        h = self.fc_class(h)
        return h

class fNIRS_VisionTransformer(MAE):
    """Classification wrapper for fNIRS-only inputs (similar to EEG wrapper)."""
    def __init__(self, n_class, embed_dim=768, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(fNIRS_VisionTransformer, self).__init__()

        self.pos_drop = nn.Dropout(p=0.5)
        self.fc_norm = norm_layer(embed_dim)
        self.fc_class = nn.Linear(embed_dim, n_class)

    def forward_feature(self, x):
        B = x.shape[0]
        x = self.fnirs_model.patch_embed(x)

        cls_tokens = self.fnirs_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.fnirs_model.pos_embed
        x = self.pos_drop(x)

        for blk in self.fnirs_model.blocks:
            x = blk(x)

        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        return x

    def forward_features(self, x):
        h = self.forward_feature(x)
        h = self.fc_norm(h)
        h = self.fc_class(h)
        return h


class EF_VisionTransformer(MAE):
    """
    Classification wrapper for multimodal inputs (EEG + fNIRS).
    Encodes both modalities and sums their pooled features before classification.
    """
    def __init__(self, n_class, embed_dim=768, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(EF_VisionTransformer, self).__init__()

        self.pos_drop = nn.Dropout(p=0.5)
        self.fc_norm = norm_layer(embed_dim)
        self.fc_class = nn.Linear(embed_dim, n_class)

    def forward_feature(self, eeg, fnirs):
        B = eeg.shape[0]
        x = self.eeg_model.patch_embed(eeg)
        y = self.fnirs_model.patch_embed(fnirs)

        cls_tokens1 = self.eeg_model.cls_token.expand(B, -1, -1)
        cls_tokens2 = self.fnirs_model.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens1, x), dim=1)
        y = torch.cat((cls_tokens2, y), dim=1)

        x = x + self.eeg_model.pos_embed
        y = y + self.fnirs_model.pos_embed

        x = self.pos_drop(x)
        y = self.pos_drop(y)

        for blk in self.eeg_model.blocks:
            x = blk(x)

        for blk in self.fnirs_model.blocks:
            y = blk(y)

        x = x[:, 1:, :].mean(dim=1)
        y = y[:, 1:, :].mean(dim=1)

        return x + y

    def forward_features(self, x, y):
        h = self.forward_feature(x, y)
        h = self.fc_norm(h)
        h = self.fc_class(h)
        return h


def vit_base(n_class=2, mode='e'):
    """Factory returning the appropriate VisionTransformer wrapper."""
    if mode == 'e':
        model = EEG_VisionTransformer(n_class=n_class)
    elif mode == 'f':
        model = fNIRS_VisionTransformer(n_class=n_class)
    elif mode == 'ef':
        model = EF_VisionTransformer(n_class=n_class)
    else:
        raise NotImplementedError('mode must be one of e/f/ef')
    return model