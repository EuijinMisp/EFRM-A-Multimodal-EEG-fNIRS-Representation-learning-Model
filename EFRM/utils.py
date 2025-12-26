# code/utils.py
"""
Utility functions used across the EFRM codebase:
- augmentations for EEG and fNIRS
- deterministic initialization and distributed helpers
- simple metrics helpers
- position embedding interpolation helper used when loading pretrained checkpoints
"""
import torch
import os
import numpy as np
import re
import torch.distributed as dist
import random
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import torch
import torch.linalg as linalg
import torch.nn as nn
import math

def eeg_noise_generator(num_points=1024):
    """Generate synthetic EEG-like noise signal for augmentation."""
    x = np.linspace(0, 1, num_points)
    intensity = np.zeros_like(x)
    for _ in range(np.random.randint(1, 10, (1,)).item()):
        amplitude = np.random.rand(1).item() * 2 - 1
        frequency = np.random.rand(1).item() * 99.5 + 0.5
        phase = np.random.rand(1).item() * 2 - 1
        intensity += amplitude * np.sin(frequency * x + phase)
    intensity = ((intensity - np.min(intensity)) * 2 / (np.max(intensity) - np.min(intensity))) - 1
    return intensity

def eeg_augmentation(x, max_noise_ratio=0.5, min_noise_ratio=0.1):
    """Add synthetic noise to EEG batch x (expected shape: [B, 1, ch, time])."""
    random_noise_ratio = (np.random.rand(1).item() * (max_noise_ratio - min_noise_ratio)) + min_noise_ratio
    random_signal_amplitude = np.random.rand(1).item() * 0.5
    noise_data = eeg_noise_generator(x.shape[-1])
    noise_data = noise_data[np.newaxis, np.newaxis, np.newaxis, :]
    noise_data = np.tile(noise_data, (x.shape[0], 1, x.shape[2], 1))
    x = (1 - random_signal_amplitude) * x + random_noise_ratio * noise_data
    return x

def fnirs_noise_generator(num_points=128):
    """Generate synthetic fNIRS-like noise signal for augmentation."""
    x = np.linspace(0, 1, num_points)
    intensity = np.zeros_like(x)
    for _ in range(np.random.randint(1, 10, (1,)).item()):
        amplitude = np.random.rand(1).item() * 0.5 - 0.25
        frequency = np.random.rand(1).item() * 0.49 + 0.01
        phase = (np.random.rand(1).item() * 0.5) - 1
        intensity += amplitude * np.sin(frequency * x + phase)
    intensity = ((intensity - np.min(intensity)) * 2 / (np.max(intensity) - np.min(intensity))) - 1
    return intensity

def fnirs_augmentation(x, max_noise_ratio=0.5, min_noise_ratio=0.1):
    """Add synthetic noise to fNIRS batch x (expected shape: [B, 2, ch, time])."""
    random_noise_ratio = (np.random.rand(1).item() * (max_noise_ratio - min_noise_ratio)) + min_noise_ratio
    random_signal_amplitude = np.random.rand(1).item() * 0.5
    noise_data = fnirs_noise_generator(x.shape[-1])
    noise_data = noise_data[np.newaxis, np.newaxis, np.newaxis, :]
    noise_data = np.tile(noise_data, (x.shape[0], 2, x.shape[2], 1))
    x = (1 - random_signal_amplitude) * x + random_noise_ratio * noise_data
    return x

def interpolate_pos_embed(model, checkpoint_model):
    """
    Interpolate the positional embedding from a checkpoint to a model with a
    different grid size. This function mutates checkpoint_model to contain
    the new pos_embed if interpolation was needed.
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def initialization(seed):
    """
    Deterministic initialization for reproducibility.
    Note: torch.use_deterministic_algorithms(True) may raise errors on older PyTorch.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)

def setup(rank, world_size, port):
    """Initialize distributed process group (NCCL backend)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroy distributed process group."""
    dist.destroy_process_group()

def measures(result, mode):
    """Compute precision/recall/f1 (macro) and their average for results dict."""
    pred = result['pred']
    label = result['label']
    precision_score, recall_score, f1_score, _ = precision_recall_fscore_support(
        np.array(label), np.array(pred), average='macro'
    )
    avg = (precision_score + recall_score + f1_score) / 3
    return {f'{mode} precision': precision_score, f'{mode} recall': recall_score,
            f'{mode} f1': f1_score, f'{mode} average': avg}

def make_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """Human-friendly sorting key for strings containing numbers."""
    return [atoi(c) for c in re.split('(\d+)', text)]