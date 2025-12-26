# code/log.py
"""
WandB logging helper for EFRM.

Contains simple wrappers for initializing wandb runs and saving small metadata needed
for resume functionality.
"""
import wandb
import torch
import numpy as np
from torchvision.utils import make_grid
import torch.nn.functional as F
import json
import os

def init_wandb(model_name, name_run, args, model_G, resume):
    """
    Initialize or resume a wandb run.

    Args:
        model_name: project name on wandb
        name_run: run name
        args: argparse namespace (will be stored in wandb.config)
        model_G: model to watch on wandb
        resume: boolean or resume id
    """
    if resume:
        wandb.init(dir='../', project=model_name, resume=resume)
    else:
        run_id = wandb.util.generate_id()
        wandb.init(dir='../', project=model_name, id=run_id, resume=resume)
        write_json(run_id)

    wandb.run.name = name_run
    wandb.config.update(args)
    wandb.watch(model_G)
    return wandb

def write_json(run_id):
    """Write a simple JSON file that stores the last run id (used for resume)."""
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    file_dir = os.path.join(parent, 'wandb')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, 'wandb-resume.json')
    data = {"run_id": run_id}
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

def save_ckp_wandb(ckp_path):
    """Upload a checkpoint to wandb (wraps wandb.save)."""
    wandb.save(ckp_path)