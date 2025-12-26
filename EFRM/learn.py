# code/learn.py
"""
Training and evaluation solver for EFRM.

This module encapsulates pretraining, fine-tuning, linear-probing, and testing logic
through the solver class. Kept behavior while updating project naming to "EFRM".
"""
import os
import copy
import torch
import torch.optim as optim
from utils import *
from dataloader import get_loader
from tqdm import tqdm
from model_pretrain import mae_vit_base
from model_finetune import vit_base
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import wandb
import log
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from timm.models.layers import trunc_normal_

class solver(object):
    """
    Solver class that configures models, optimizers, schedulers, and runs training loops.
    """
    def __init__(self, args, retrain):
        torch.cuda.set_device(args.gpu_id)
        self.mode = args.mode
        self.gpu_id = args.gpu_id
        self.rp_gpu_id = args.rp_gpu_id
        self.retrain = retrain
        self.multigpu_use = args.multigpu_use
        self.run_name = args.run_name
        self.model_name = args.model_name

        if self.gpu_id == self.rp_gpu_id:
            print('-----------------------')
            print(f'Selected Model and Defined Hyper-Parameters: {args}')

    def pretrain_settting(self, args):
        """Set up the pretraining model, optimizer, scheduler, and data loader."""
        if self.model_name == 'EFMF':
            self.model = mae_vit_base(eeg_size=args.target_eeg_size, fnirs_size=args.target_fnirs_size, mask_ratio=args.mask_ratio).to(self.gpu_id)
        else:
            raise NotImplementedError('please select model in [EFMF]')

        for _, p in self.model.named_parameters():
            p.requires_grad = True

        if self.gpu_id == self.rp_gpu_id:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_non_parameters = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            print('-----------------------')
            print(f'number of training params (M):{n_parameters/1e6}')
            print(f'number of non training params (M):{n_non_parameters/1e6}')
            print('-----------------------')

        if self.multigpu_use:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.pre_lr, betas=(0.9, 0.95))
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=int(args.pre_niter / 3),
            cycle_mult=1, max_lr=args.pre_lr, min_lr=0, warmup_steps=0, gamma=0.7
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.pretrain_save_path = os.path.join(args.pre_model_save_path, self.run_name)

        if self.gpu_id == self.rp_gpu_id:
            make_directory(self.pretrain_save_path)

            if self.retrain:
                # Use unified project name "EFRM" for wandb
                log.init_wandb('EFRM', self.run_name, args, self.model, resume=True)
                self.load_checkpoint(os.path.join(args.load_model_path, self.run_name, 'last_model'))
            else:
                log.init_wandb('EFRM', self.run_name, args, self.model, resume=False)

        self.pretrain_data_loader = get_loader(args, 'pretrain', args.batch_size)

    def finetune_settting(self, args):
        """Set up the fine-tuning model, optimizer, scheduler, loaders, and wandb logging."""
        self.n_class = args.n_class
        if self.model_name.split('_')[0] == 'EFMF':
            if args.target_dataset_type == 'sleepstage_eeg':
                self.model = vit_base(n_class=args.n_class, mode='e').to(self.gpu_id)
            elif args.target_dataset_type == 'mental_arithmetic_fnirs':
                self.model = vit_base(n_class=args.n_class, mode='f').to(self.gpu_id)
            elif args.target_dataset_type == 'drowsiness_eeg':
                self.model = vit_base(n_class=args.n_class, mode='e').to(self.gpu_id)
            elif args.target_dataset_type == 'drowsiness_fnirs':
                self.model = vit_base(n_class=args.n_class, mode='f').to(self.gpu_id)
            elif args.target_dataset_type == 'drowsiness_multi':
                self.model = vit_base(n_class=args.n_class, mode='ef').to(self.gpu_id)
            else:
                raise NotImplementedError('please select correct target_dataset_type')
        else:
            raise NotImplementedError('please select model in [EFMF]')

        # Load pretrained weights (from pretraining)
        self.load_checkpoint(os.path.join(args.pretrained_model_path, self.model_name, 'last_model'))

        # Wrap classification head with BatchNorm as in original repo
        self.model.fc_class = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.fc_class.in_features, affine=False, eps=1e-6),
            self.model.fc_class
        ).to(self.gpu_id)

        for _, p in self.model.named_parameters():
            p.requires_grad = True

        if self.gpu_id == self.rp_gpu_id:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_non_parameters = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            print('-----------------------')
            print(f'number of training params (M):{n_parameters/1e6}')
            print(f'number of non training params:{n_non_parameters}')
            print('-----------------------')

        if self.multigpu_use:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.fine_lr, betas=(0.9, 0.95))
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer, first_cycle_steps=int(args.fine_niter),
            cycle_mult=1, max_lr=args.fine_lr, min_lr=1e-6, warmup_steps=0, gamma=0.7
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_step = int(args.fine_niter / 5)
        self.finetune_save_path = os.path.join(args.fine_model_save_path, args.target_dataset_type, self.run_name)

        if self.gpu_id == self.rp_gpu_id:
            make_directory(self.finetune_save_path)
            # Use args.project_name where available (keeps dataset specific names)
            log.init_wandb(args.project_name, self.run_name + '_' + args.mode, args, self.model, resume=False)

        self.finetune_data_loader = get_loader(args, 'finetune', args.batch_size)
        self.test_data_loader = get_loader(args, 'test')

    def linprobe_settting(self, args):
        """Set up the model for linear probing: freeze backbone, train only head."""
        self.n_class = args.n_class
        if self.model_name.split('_')[0] == 'EFMF':
            if args.target_dataset_type == 'sleepstage_eeg':
                self.model = vit_base(n_class=args.n_class, mode='e').to(self.gpu_id)
            elif args.target_dataset_type == 'mental_arithmetic_fnirs':
                self.model = vit_base(n_class=args.n_class, mode='f').to(self.gpu_id)
            elif args.target_dataset_type == 'drowsiness_eeg':
                self.model = vit_base(n_class=args.n_class, mode='e').to(self.gpu_id)
            elif args.target_dataset_type == 'drowsiness_fnirs':
                self.model = vit_base(n_class=args.n_class, mode='f').to(self.gpu_id)
            elif args.target_dataset_type == 'drowsiness_multi':
                self.model = vit_base(n_class=args.n_class, mode='ef').to(self.gpu_id)
            else:
                raise NotImplementedError('please select correct target_dataset_type')
        else:
            raise NotImplementedError('please select model in [EFMF]')

        self.load_checkpoint(os.path.join(args.pretrained_model_path, self.model_name, 'last_model'))
        self.model.fc_class = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.model.fc_class.in_features, affine=False, eps=1e-6),
            self.model.fc_class
        ).to(self.gpu_id)

        # freeze backbone parameters
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        for _, p in self.model.fc_class.named_parameters():
            p.requires_grad = True

        if self.gpu_id == self.rp_gpu_id:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_non_parameters = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            print('-----------------------')
            print(f'number of training params :{n_parameters}')
            print(f'number of non training params (M):{n_non_parameters/1e6}')
            print('-----------------------')

        if self.multigpu_use:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.optimizer = optim.AdamW(self.model.fc_class.parameters(), lr=args.fine_lr, betas=(0.9, 0.95))
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer, first_cycle_steps=int(args.fine_niter),
            cycle_mult=1, max_lr=args.fine_lr, min_lr=1e-6, warmup_steps=0, gamma=0.7
        )

        self.scaler = torch.cuda.amp.GradScaler()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_step = int(args.fine_niter / 5)
        self.finetune_save_path = os.path.join(args.fine_model_save_path, args.target_dataset_type, self.run_name)

        if self.gpu_id == self.rp_gpu_id:
            make_directory(self.finetune_save_path)
            log.init_wandb(args.project_name, self.run_name + '_' + args.mode, args, self.model, resume=False)

        self.finetune_data_loader = get_loader(args, 'linprobe', args.batch_size)
        self.test_data_loader = get_loader(args, 'test')

    def prediction(self, data_loader, slice_size=16):
        """Run prediction on a dataloader and collect labels and predictions."""
        pred_list = []
        label_list = []
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                if idx > 10:
                    break

                # Single-modality (x) or multi-modality (x1, x2) data format handling
                if len(data) == 2:
                    x_full = data['x'][0].to(self.gpu_id)
                    y_full = data['y'][0]
                    total_samples = x_full.size(0)
                    for start_idx in range(0, total_samples, slice_size):
                        end_idx = min(start_idx + slice_size, total_samples)
                        x = x_full[start_idx:end_idx]
                        y = y_full[start_idx:end_idx]

                        if self.multigpu_use:
                            pred = self.model.module.forward_features(x)
                        else:
                            pred = self.model.forward_features(x)
                        pred = torch.argmax(pred, dim=1).cpu().tolist()
                        y = y.tolist()
                        pred_list += pred
                        label_list += y

                elif len(data) == 3:
                    x1_full = data['x1'][0].to(self.gpu_id)
                    x2_full = data['x2'][0].to(self.gpu_id)
                    y_full = data['y'][0]
                    total_samples = x1_full.size(0)
                    for start_idx in range(0, total_samples, slice_size):
                        end_idx = min(start_idx + slice_size, total_samples)
                        x1 = x1_full[start_idx:end_idx]
                        x2 = x2_full[start_idx:end_idx]
                        y = y_full[start_idx:end_idx]

                        if self.multigpu_use:
                            pred = self.model.module.forward_features(x1, x2)
                        else:
                            pred = self.model.forward_features(x1, x2)
                        pred = torch.argmax(pred, dim=1).cpu().tolist()
                        y = y.tolist()
                        pred_list += pred
                        label_list += y

        return {'pred': pred_list, 'label': label_list}

    def pre_train(self, args):
        """Main pretraining loop."""
        self.pretrain_settting(args)

        if self.gpu_id == self.rp_gpu_id:
            print('-----------------------')
            print('Pre-training mode start')
            print('-----------------------')

        for i in range(args.pre_niter):
            self.global_iter = i + 1
            acc_total_loss = 0
            acc_eeg_recon_loss = 0
            acc_fnirs_recon_loss = 0
            acc_clip_loss = 0

            if self.gpu_id == self.rp_gpu_id:
                # Note: scheduler.get_lr() may be deprecated in some torch versions;
                # original code calls it — kept for compatibility. Replace with
                # scheduler.get_last_lr() if desired.
                print('current learning rate:%s' % str(self.scheduler.get_lr()))
                progress_bar = tqdm(self.pretrain_data_loader)
            else:
                progress_bar = self.pretrain_data_loader

            for idx, data in enumerate(progress_bar):
                x = data['eeg'][0].to(self.gpu_id)
                y = data['fnirs'][0].to(self.gpu_id)
                px = data['pair_eeg'][0].to(self.gpu_id)
                py = data['pair_fnirs'][0].to(self.gpu_id)

                with torch.cuda.amp.autocast():
                    eeg_recon_loss, fnirs_recon_loss, clip_loss = self.model(x, y, px, py)
                    loss = eeg_recon_loss + fnirs_recon_loss + clip_loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                acc_total_loss += loss.item()
                acc_eeg_recon_loss += eeg_recon_loss.item()
                acc_fnirs_recon_loss += fnirs_recon_loss.item()
                acc_clip_loss += clip_loss.item()
            self.scheduler.step()

            if self.gpu_id == self.rp_gpu_id:
                self.save_checkpoint(self.pretrain_save_path, 'last_model')
                print(f'Training step:{self.global_iter}/{args.pre_niter}, Loss:{acc_total_loss:.4f}')
                wandb.log({'Total loss': acc_total_loss}, step=self.global_iter)
                wandb.log({'EEG reconstruction loss': acc_eeg_recon_loss}, step=self.global_iter)
                wandb.log({'fNIRSreconstruction loss': acc_fnirs_recon_loss}, step=self.global_iter)
                wandb.log({'Clip loss': acc_clip_loss}, step=self.global_iter)

    def fine_tune(self, args):
        """Main fine-tuning / linear probing loop (shared entry, switches above)."""
        if args.mode == 'linprobe':
            self.linprobe_settting(args)
        elif args.mode == 'finetune':
            self.finetune_settting(args)

        if (self.gpu_id == self.rp_gpu_id) and (args.mode == 'finetune'):
            print('-----------------------')
            print('Fine-tuning mode start')
            print('-----------------------')
            best_score = 0
        elif (self.gpu_id == self.rp_gpu_id) and (args.mode == 'linprobe'):
            print('-----------------------')
            print('Linear-probing mode start')
            print('-----------------------')
            best_score = 0

        for i in range(args.fine_niter):
            self.global_iter = i + 1
            acc_loss = 0

            if self.gpu_id == self.rp_gpu_id:
                print('current learning rate:%s' % str(self.scheduler.get_lr()))
                progress_bar = tqdm(self.finetune_data_loader)
            else:
                progress_bar = self.finetune_data_loader

            for idx, data in enumerate(progress_bar):
                if len(data) == 2:
                    x = data['x'][0].to(self.gpu_id)
                    y = data['y'][0].to(self.gpu_id)
                    with torch.cuda.amp.autocast():
                        if self.multigpu_use:
                            pred = self.model.module.forward_features(x)
                        else:
                            pred = self.model.forward_features(x)
                        loss = self.criterion(pred, y)
                elif len(data) == 3:
                    x1 = data['x1'][0].to(self.gpu_id)
                    x2 = data['x2'][0].to(self.gpu_id)
                    y = data['y'][0].to(self.gpu_id)
                    with torch.cuda.amp.autocast():
                        if self.multigpu_use:
                            pred = self.model.module.forward_features(x1, x2)
                        else:
                            pred = self.model.forward_features(x1, x2)
                        loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                acc_loss += loss.item()

            self.scheduler.step()

            if self.gpu_id == self.rp_gpu_id:
                print(f'Training step:{self.global_iter}/{args.fine_niter}, Loss:{acc_loss:.4f}')
                self.save_checkpoint(self.finetune_save_path, 'last_model')
                wandb.log({'Training loss': acc_loss}, step=self.global_iter)

                if (i % self.save_step) == 0:
                    torch.cuda.empty_cache()
                    results = self.prediction(self.test_data_loader)

                    self.evaluate_results = measures(results, 'test')
                    wandb.log(self.evaluate_results, step=self.global_iter)
                    score = self.evaluate_results['test average']

                    if best_score < score:
                        best_score = score
                        self.save_checkpoint(self.finetune_save_path, 'best_model')
                        print('The best model is saved')

    def test(self, args):
        """Test entrypoint: create save dir, load finetuned model, and run prediction."""
        make_directory(args.test_save)
        self.test_data_loader = get_loader(args, 'test')
        self.load_checkpoint(args.finetuned_model_path)
        self.prediction(self.test_data_loader)

    def save_checkpoint(self, save_path, filename, silent=True):
        """Save checkpoint dict with optimizer/scheduler/model states."""
        states = {'iter': self.global_iter,
                  'optim_states': self.optimizer.state_dict(),
                  'lr_states': self.scheduler.state_dict()}

        if self.multigpu_use:
            states['model_states'] = self.model.module.state_dict()
        else:
            states['model_states'] = self.model.state_dict()

        file_path = os.path.join(save_path, filename)
        torch.save(states, file_path)

        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, file_path):
        """Load checkpoint and restore model/optimizer/scheduler depending on mode."""
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path, map_location='cpu')

            if self.mode == 'pretrain':
                self.global_iter = checkpoint['iter']
                self.optimizer.load_state_dict(checkpoint['optim_states'])
                self.scheduler.load_state_dict(checkpoint['lr_states'])
                self.model.load_state_dict(checkpoint['model_states'])
                print("=> loaded checkpoint from:'{} (iter {})'".format(file_path, self.global_iter))

            elif self.mode in ('finetune', 'linprobe'):
                checkpoint_model = checkpoint['model_states']
                state_dict = self.model.state_dict()

                # If classifier head shapes differ, remove those keys from pretrained checkpoint
                for k in ['fc_norm.bias', 'fc_norm.weight', 'fc_class.bias', 'fc_class.weight']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                interpolate_pos_embed(self.model, checkpoint_model)  # interpolate position embedding
                msg = self.model.load_state_dict(checkpoint_model, strict=False)  # load pre-trained model
                assert set(msg.missing_keys) == {'fc_norm.bias', 'fc_norm.weight', 'fc_class.bias', 'fc_class.weight'}

                trunc_normal_(self.model.fc_class.weight, std=2e-5)  # manually initialize fc layer
                print("=> loaded pretrained checkpoint from:'{}'".format(file_path))

            elif self.mode == 'test':
                self.model.load_state_dict(checkpoint['model_states'])
                print("=> loaded finetuned checkpoint from:'{}'".format(file_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(file_path))