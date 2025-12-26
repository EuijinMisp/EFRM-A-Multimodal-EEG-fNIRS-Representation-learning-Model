# code/main_linearprobe.py
"""
Entrypoint for linear-probing experiments (train only head) for EFRM.
Mostly the same structure as main_finetune.
"""
from __future__ import print_function
import argparse
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch.backends.cudnn as cudnn
from learn import solver
import numpy as np
import random
from utils import setup, cleanup, initialization
import torch.multiprocessing as mp
import torch.distributed as dist
from dataloader import dataset_option

def main():
    parser = argparse.ArgumentParser(description='PyTorch EFRM: EEG & fNIRS Multi Modal Foundation model')

    ################################### Common Parameters ###################################
    parser.add_argument('--mode', default='linprobe', help='linprobe')
    parser.add_argument('--run_name', default='EFMF_1shot_2c')
    parser.add_argument('--model_name', default='EFMF', help='EFMF')
    parser.add_argument('--gpu_ids', type=list, default=[0], help='number of GPUs to use')
    parser.add_argument('--target_dataset_type', type=str, default='sleepstage_eeg',
                        metavar='N', help='sleepstage_eeg, mental_arithmetic_fnirs, drowsiness_eeg, drowsiness_fnirs, drowsiness_multi')
    parser.add_argument('--k_shot', type=int, default=1, metavar='N', help='the number of shots for training')
    parser.add_argument('--n_class', type=int, default=2, metavar='N', help='2, 3')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='16')
    parser.add_argument('--port', type=int, default=12355, metavar='N', help='port')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')

    ################################### Training Parameters ###################################
    parser.add_argument('--pretrained_model_path', default='../run/pretrain', help='pretrained_model_path')
    parser.add_argument('--fine_model_save_path', default='../run/linprobe', help='fine_model_save_path')
    parser.add_argument('--fine_lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.00001)')
    parser.add_argument('--fine_niter', type=int, default=200, metavar='N', help='number of epochs to train (default: 50)')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    args = dataset_option(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in args.gpu_ids)

    if (type(args.gpu_ids) == list) and len(args.gpu_ids) > 0:
        args.world_size = len(args.gpu_ids)
    else:
        args.world_size = torch.cuda.device_count()
        if args.world_size == 0:
            raise RuntimeError('GPU is not available')

    args.workers = int(args.workers * args.world_size)
    args.batch_size = int(args.batch_size / args.world_size)
    args.multigpu_use = args.world_size > 1
    args.rp_gpu_id = 0
    print(f'The number of GPUs for multiprocessing:{args.world_size}')

    if args.multigpu_use:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        main_worker(0, args)


def main_worker(rank, args):
    """Per-process worker for distributed training or single-GPU training."""
    print(f"Used GPU ID:{args.gpu_ids[rank]}")
    initialization(args.seed)
    args.gpu_id = rank

    if args.multigpu_use:
        setup(rank, args.world_size, args.port)

    net = solver(args, retrain=False)

    if args.multigpu_use:
        dist.barrier()
        net.fine_tune(args)
        cleanup()
    else:
        net.fine_tune(args)

if __name__ == '__main__':
    main()