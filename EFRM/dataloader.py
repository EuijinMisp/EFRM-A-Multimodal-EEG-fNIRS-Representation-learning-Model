# code/dataloader.py
"""
Data loader module for EFRM (multimodal EEG / fNIRS).

This file contains dataset classes and a helper `get_loader` function that
constructs PyTorch DataLoader objects for pretraining, fine-tuning, linear-probing,
and testing modes.

Notes:
- Minimal changes: added docstrings and inline comments for clarity and
  updated top-level project references to "EFRM".
- Kept original class/function names to preserve compatibility with other scripts.
"""
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
from utils import *  # retain for compatibility (natural_keys, augmentations, etc.)
import numpy as np
import random

def dataset_option(args):
    """Populate dataset-related options on args (sizes and default data dirs)."""
    # Target input sizes used across datasets
    args.target_eeg_size = (24, 1024)  # 128 Hz * 8 sec = 1024 time points
    args.target_fnirs_size = (64, 128) # 16 Hz * 8 sec = 128 time points

    if args.mode == 'pretrain':
        args.pretrain_data_dir = '../data/pretrain'
    elif args.mode in ('finetune', 'linprobe', 'test'):
        # downstream dataset configuration
        if args.target_dataset_type == 'sleepstage_eeg':
            args.project_name = args.target_dataset_type
            args.target_data_dir = '../data/downstream/EEG/sleepstage'
        elif args.target_dataset_type == 'mental_arithmetic_fnirs':
            args.project_name = args.target_dataset_type
            args.target_data_dir = '../data/downstream/fNIRS/mental_arithmetic'
            args.n_class = 2
        elif args.target_dataset_type in ('drowsiness_eeg', 'drowsiness_fnirs', 'drowsiness_multi'):
            args.project_name = 'drowsiness_multi'
            args.target_data_dir = '../data/downstream/EEG-fNIRS/drowsiness'
        else:
            raise NotImplementedError(
                'select dataset_type in [sleepstage_eeg, mental_arithmetic_fnirs, '
                'drowsiness_eeg, drowsiness_fnirs, drowsiness_multi]'
            )

    return args

def pair_shuffle(data, label=None):
    """Shuffle data (and labels) along the first axis and return them."""
    indices = np.random.permutation(data.shape[0])
    data = data[indices]

    if label is not None:
        label = label[indices]
        return data, label
    else:
        return data

class Pratrain_data(data.Dataset):
    """
    Dataset class used for pretraining (original name preserved: 'Pratrain_data').

    Expected folder structure under data_dir:
      - EEG_only/
      - fNIRS_only/
      - EEG-fNIRS/  (paired examples)
    Each subfolder is iterated and .npy files are expected inside.
    """
    def __init__(self, target_eeg_size, target_fnirs_size, data_dir, mode, batch_size, gpu_id, rp_gpu_id):
        self.mode = mode
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.rp_gpu_id = rp_gpu_id
        self.data_dir = data_dir
        self.target_eeg_nch, self.target_eeg_ntime = target_eeg_size[0], target_eeg_size[1]
        self.target_fnirs_nch, self.target_fnirs_ntime = target_fnirs_size[0], target_fnirs_size[1]
        self.preprocess()

    def get_all_file_paths(self, directory):
        """Recursively collect all file paths under directory."""
        file_paths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                file_paths.append(os.path.join(root, filename))
        return file_paths

    def preprocess(self):
        """
        Build lists of available EEG-only, fNIRS-only, and paired examples.
        This function intentionally mirrors the original implementation; if data
        layout differs, adapt this method.
        """
        self.eeg_data_dir = os.path.join(self.data_dir, 'EEG_only')
        self.fnirs_data_dir = os.path.join(self.data_dir, 'fNIRS_only')
        self.pair_data_dir = os.path.join(self.data_dir, 'EEG-fNIRS')

        self.dataset_eeg = []
        self.dataset_fnirs = []
        self.dataset_pair1 = []
        self.dataset_pair2 = []

        # Defensive: if directories are missing, raise clear error
        if not os.path.isdir(self.eeg_data_dir):
            raise RuntimeError(f"EEG directory not found: {self.eeg_data_dir}")
        if not os.path.isdir(self.fnirs_data_dir):
            raise RuntimeError(f"fNIRS directory not found: {self.fnirs_data_dir}")
        if not os.path.isdir(self.pair_data_dir):
            raise RuntimeError(f"Paired directory not found: {self.pair_data_dir}")

        # Collect EEG-only files
        for nset in os.listdir(self.eeg_data_dir):
            self.dataset_eeg += self.get_all_file_paths(os.path.join(self.eeg_data_dir, nset))

        # Collect fNIRS-only files
        for nset in os.listdir(self.fnirs_data_dir):
            self.dataset_fnirs += self.get_all_file_paths(os.path.join(self.fnirs_data_dir, nset))

        # Collect paired dataset folders/files
        for nset in os.listdir(self.pair_data_dir):
            self.dataset_pair1 += self.get_all_file_paths(os.path.join(self.pair_data_dir, nset))

        # The original code reduced pair paths by selecting the first 6 path components.
        # Keep behavior but add a comment: this is brittle and can be improved.
        for path in self.dataset_pair1:
            components = path.split('/')
            if len(components) >= 6:
                reduced = os.path.join(*components[:6])
            else:
                reduced = path
            self.dataset_pair2.append(reduced)
        self.dataset_pair = list(set(self.dataset_pair2))

        if self.gpu_id == self.rp_gpu_id:
            print(f'Finished preprocessing {self.mode} dataset...')

    def data_select(self, data, mode='e'):
        """
        Select or pad channels/time windows to the target size.

        Args:
            data: numpy array representing the recording.
            mode: 'e' for EEG (assumed shape [ch, time]),
                  'f' for fNIRS (assumed shape [2, ch, time])
        """
        if mode == 'e':
            nch, ntime = data.shape
            if nch > self.target_eeg_nch:
                c_idx = np.random.randint(low=0, high=nch - self.target_eeg_nch + 1)
                t_idx = np.random.randint(low=0, high=ntime - self.target_eeg_ntime + 1)
                data = data[c_idx:c_idx + self.target_eeg_nch, t_idx:t_idx + self.target_eeg_ntime]
            elif nch == self.target_eeg_nch:
                t_idx = np.random.randint(low=0, high=ntime - self.target_eeg_ntime + 1)
                data = data[:, t_idx:t_idx + self.target_eeg_ntime]
            else:
                raise NotImplementedError('the channel of data is smaller than 24')

        elif mode == 'f':
            # fNIRS data expected shape: [2, ch, time]
            _, nch, ntime = data.shape
            if nch > self.target_fnirs_nch:
                c_idx = np.random.randint(low=0, high=nch - self.target_fnirs_nch + 1)
                t_idx = np.random.randint(low=0, high=ntime - self.target_fnirs_ntime + 1)
                data = data[:, c_idx:c_idx + self.target_fnirs_nch, t_idx:t_idx + self.target_fnirs_ntime]
            elif nch == self.target_fnirs_nch:
                t_idx = np.random.randint(low=0, high=ntime - self.target_fnirs_ntime + 1)
                data = data[:, :, t_idx:t_idx + self.target_fnirs_ntime]
            elif nch < self.target_fnirs_nch:
                # If fewer channels, repeat and possibly mirror channels until target reached
                t_idx = np.random.randint(low=0, high=ntime - self.target_fnirs_ntime + 1)
                data = data[:, :, t_idx:t_idx + self.target_fnirs_ntime]

                while data.shape[1] < self.target_fnirs_nch:
                    if np.random.rand() < 0.5:
                        data = np.concatenate((data, data), axis=1)
                    else:
                        data = np.concatenate((data, data[:, ::-1, :]), axis=1)

                if data.shape[1] > self.target_fnirs_nch:
                    data = data[:, :self.target_fnirs_nch, :]
        else:
            raise NotImplementedError('Please select the mode between "e" and "f"')

        return data

    def pair_data_select(self, data1, data2):
        """
        Select a random subset of paired examples and ensure channel sizes match
        expected target channel counts. This function returns arrays with batch
        dimension equal to self.batch_size.
        """
        indices = np.random.permutation(data1.shape[0])
        data1 = data1[indices][:self.batch_size]
        data2 = data2[indices][:self.batch_size]

        _, _, eeg_nch, _ = data1.shape
        c_idx = np.random.randint(low=0, high=eeg_nch - self.target_eeg_nch + 1)
        data1 = data1[:, :, c_idx:c_idx + self.target_eeg_nch, :]

        # Expand fNIRS channels if needed (repeat/mirror)
        while data2.shape[2] < self.target_fnirs_nch:
            if np.random.rand() < 0.5:
                data2 = np.concatenate((data2, data2), axis=2)
            else:
                data2 = np.concatenate((data2, data2[:, :, ::-1, :]), axis=2)

        if data2.shape[2] > self.target_fnirs_nch:
            data2 = data2[:, :, :self.target_fnirs_nch, :]

        return data1, data2

    def __getitem__(self, index):
        """Return a single pretraining batch (as dict) built from random selections."""
        index_eeg = index % len(self.dataset_eeg)
        index_fnirs = index % len(self.dataset_fnirs)
        index_pair = index % len(self.dataset_pair)

        eeg_fname = self.dataset_eeg[index_eeg]
        fnirs_fname = self.dataset_fnirs[index_fnirs]
        pair_fname = self.dataset_pair[index_pair]

        out_eeg = []
        out_fnirs = []
        out_pair_eeg = []
        out_pair_fnirs = []

        # Load raw numpy arrays; shapes expected by other parts of the code:
        # eeg: [ch, time], fnirs: [2, ch, time]
        eeg = np.load(eeg_fname)                                   # [ch, time]
        fnirs = np.load(fnirs_fname)                               # [2, ch, time]
        pair_eeg = np.load(os.path.join(pair_fname, 'eeg.npy'))    # [B, 1, nch, ntime]
        pair_fnirs = np.load(os.path.join(pair_fname, 'fnirs.npy'))# [B, 2, nch, ntime]

        # Build a batch by sampling and cropping/padding as needed
        for _ in range(self.batch_size):
            selected_data = self.data_select(eeg, mode='e')
            out_eeg.append(selected_data)
        eeg = np.array(out_eeg)[:, np.newaxis, :, :]  # shape: [B, 1, nch, ntime]

        for _ in range(self.batch_size):
            selected_data = self.data_select(fnirs, mode='f')
            out_fnirs.append(selected_data)
        fnirs = np.array(out_fnirs)  # shape: [B, 2, nch, ntime]

        pair_eeg, pair_fnirs = self.pair_data_select(pair_eeg, pair_fnirs)

        # Random augmentations (augmentations implemented in utils)
        if np.random.rand() > 0.5:
            eeg = eeg_augmentation(eeg)
            fnirs = fnirs_augmentation(fnirs)
            pair_eeg = eeg_augmentation(pair_eeg)
            pair_fnirs = fnirs_augmentation(pair_fnirs)

        # Convert to torch tensors
        eeg = torch.tensor(eeg, dtype=torch.float32)
        fnirs = torch.tensor(fnirs, dtype=torch.float32)
        pair_eeg = torch.tensor(pair_eeg, dtype=torch.float32)
        pair_fnirs = torch.tensor(pair_fnirs, dtype=torch.float32)
        return {'eeg': eeg, 'fnirs': fnirs, 'pair_eeg': pair_eeg, 'pair_fnirs': pair_fnirs}

    def __len__(self):
        """Return an upper bound on samples for iteration (largest dataset length)."""
        return max(len(self.dataset_eeg), len(self.dataset_fnirs), len(self.dataset_pair))


class SleepStage(data.Dataset):
    """
    Dataset class for sleep staging (EEG). Supports k-shot fine-tuning and test modes.
    """
    def __init__(self, target_data_shape, k_shot, n_class, data_dir, mode, gpu_id, rp_gpu_id):
        self.mode = mode
        self.target_data_shape = target_data_shape
        self.k_shot = k_shot
        self.n_class = n_class
        self.gpu_id = gpu_id
        self.rp_gpu_id = rp_gpu_id
        self.data_dir = data_dir

        # default test sizes are dataset-specific
        if self.n_class == 2:
            self.n_test = 70
        elif self.n_class == 3:
            self.n_test = 14
        self.preprocess()

    def preprocess(self):
        """Collect dataset list and choose train/test splits (deterministic ordering)."""
        self.data_list = os.listdir(self.data_dir)
        self.data_list.sort(key=natural_keys)
        self.n_train_dataset = len(self.data_list) - self.n_test

        if self.mode in ('finetune', 'linprobe'):
            if self.k_shot > self.n_train_dataset:
                raise NotImplementedError('please select k_shot <= n_train_dataset(64)')
            else:
                self.dataset = self.data_list[:self.k_shot]
        elif self.mode == 'test':
            self.dataset = self.data_list[self.n_train_dataset:]

        if self.gpu_id == self.rp_gpu_id:
            print(f'Finished preprocessing {self.mode} dataset...')

        return self.dataset

    def channel_matching(self, data):
        """If data has fewer channels than target, repeat channels to match shape."""
        ncopy_ch = self.target_data_shape[0] - data.shape[1]

        if ncopy_ch > 0:
            repeats = (ncopy_ch // data.shape[1]) + 1
            extended_data = np.tile(data, (1, repeats, 1))
            reshaped_data = np.concatenate((data, extended_data[:, :ncopy_ch, :]), axis=1)
            return reshaped_data
        return data

    def data_labeing_3c(self, awake, nonrem, rem):
        """Concatenate per-class arrays and produce labels for 3-class sleep staging."""
        data = np.concatenate((awake, nonrem, rem), axis=0)
        label = []
        label += [0] * awake.shape[0]
        label += [1] * nonrem.shape[0]
        label += [2] * rem.shape[0]
        label = np.array(label)

        if self.mode in ('finetune', 'linprobe'):
            indices = np.random.permutation(data.shape[0])
            data = data[indices]
            label = label[indices]

        return data, label

    def data_labeing_2c(self, awake, sleep):
        """Concatenate two classes and produce labels for 2-class sleep staging."""
        data = np.concatenate((awake, sleep), axis=0)
        label = []
        label += [0] * awake.shape[0]
        label += [1] * sleep.shape[0]
        label = np.array(label)

        if self.mode in ('finetune', 'linprobe'):
            indices = np.random.permutation(data.shape[0])
            data = data[indices]
            label = label[indices]

        return data, label

    def __getitem__(self, index):
        """Return sample and label for a particular subject/dataset index."""
        awake_fname = os.path.join(self.data_dir, self.dataset[index], 'awake.npy')
        nonrem_fname = os.path.join(self.data_dir, self.dataset[index], 'nonrem.npy')
        rem_fname = os.path.join(self.data_dir, self.dataset[index], 'rem.npy')

        awake = np.load(awake_fname)
        nonrem = np.load(nonrem_fname)
        rem = np.load(rem_fname)
        min_batch = min(awake.shape[0], nonrem.shape[0], rem.shape[0])
        start = int((index / len(self.dataset)) * min_batch)

        if self.n_class == 3:
            if self.mode in ('finetune', 'linprobe'):
                awake = awake[start:start + 1]
                nonrem = nonrem[start:start + 1]
                rem = rem[start:start + 1]
            elif self.mode == 'test':
                awake = awake[:64]
                nonrem = nonrem[:64]
                rem = rem[:64]

            data, label = self.data_labeing_3c(awake, nonrem, rem)

        elif self.n_class == 2:
            if self.mode in ('finetune', 'linprobe'):
                awake = awake[start:start + 1]
                if index % 2 == 0:
                    sleep = nonrem[start:start + 1]
                else:
                    sleep = rem[start:start + 1]
            elif self.mode == 'test':
                awake = awake[:64]
                nonrem = nonrem[:32]
                rem = rem[:32]
                sleep = np.concatenate((nonrem, rem), axis=0)

            data, label = self.data_labeing_2c(awake, sleep)
        else:
            raise NotImplementedError('please select n_class==2 or n_class==3')

        data = self.channel_matching(data)[:, np.newaxis, :, :]
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)
        return {'x': data, 'y': label}

    def __len__(self):
        return len(self.dataset)


class Mental_Arithmetic(data.Dataset):
    """Dataset class for the mental arithmetic fNIRS downstream task."""
    def __init__(self, target_data_shape, k_shot, data_dir, mode, gpu_id, rp_gpu_id):
        self.mode = mode
        self.target_data_shape = target_data_shape
        self.k_shot = k_shot
        self.gpu_id = gpu_id
        self.rp_gpu_id = rp_gpu_id
        self.data_dir = data_dir
        self.n_test = 4
        self.d_per_sub = 2
        self.n_train_dataset = int(self.k_shot / self.d_per_sub)
        self.preprocess()

    def preprocess(self):
        self.data_list = os.listdir(self.data_dir)
        self.data_list.sort(key=natural_keys)
        self.n_train_dataset_total = len(self.data_list) - self.n_test

        if self.mode in ('finetune', 'linprobe'):
            if self.n_train_dataset > self.n_train_dataset_total:
                raise NotImplementedError('please select n_train_dataset <= n_train_dataset_total(4)')
            else:
                self.dataset = self.data_list[:self.n_train_dataset]
        elif self.mode == 'test':
            self.dataset = self.data_list[self.n_train_dataset_total:]

        if self.gpu_id == self.rp_gpu_id:
            print(f'Finished preprocessing {self.mode} dataset...')

        return self.dataset

    def data_labeing(self, classA, classB):
        data = np.concatenate((classA, classB), axis=0)
        label = [0] * classA.shape[0] + [1] * classB.shape[0]
        label = np.array(label)

        if self.mode in ('finetune', 'linprobe'):
            indices = np.random.permutation(data.shape[0])
            data = data[indices]
            label = label[indices]

        return data, label

    def __getitem__(self, index):
        sid = self.dataset[index]

        bl_fname = os.path.join(self.data_dir, sid, 'bl.npy')
        ma_fname = os.path.join(self.data_dir, sid, 'ma.npy')

        if self.mode in ('finetune', 'linprobe'):
            bl = np.load(bl_fname)
            ma = np.load(ma_fname)
            bl = np.array([bl[0, :, :, :], bl[-1, :, :, :]])
            ma = np.array([ma[0, :, :, :], ma[-1, :, :, :]])
        elif self.mode == 'test':
            bl = np.load(bl_fname)[:64]
            ma = np.load(ma_fname)[:64]

        data, label = self.data_labeing(bl, ma)

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        residual_nch = self.target_data_shape[0] - data.shape[2]
        if residual_nch > 0:
            data = torch.cat([data, data[:, :, :residual_nch, :]], axis=2)

        return {'x': data, 'y': label}

    def __len__(self):
        return len(self.dataset)


class Drowsiness(data.Dataset):
    """Dataset class for drowsiness detection across EEG/fNIRS/multi modalities."""
    def __init__(self, modality, target_eeg_shape, target_fnirs_shape, k_shot, n_class, data_dir, mode, gpu_id, rp_gpu_id):
        self.mode = mode
        self.modality = modality
        self.target_eeg_shape = target_eeg_shape
        self.target_fnirs_shape = target_fnirs_shape
        self.k_shot = k_shot
        self.n_class = n_class
        self.gpu_id = gpu_id
        self.rp_gpu_id = rp_gpu_id
        self.data_dir = data_dir
        self.n_test = 5
        self.preprocess()

    def preprocess(self):
        self.data_list = os.listdir(self.data_dir)
        self.data_list.sort(key=natural_keys)
        self.n_train_dataset = len(self.data_list) - self.n_test

        if self.mode in ('finetune', 'linprobe'):
            if self.k_shot > self.n_train_dataset:
                self.dataset = self.data_list[:self.n_train_dataset]
            else:
                self.dataset = self.data_list[:self.k_shot]
        elif self.mode == 'test':
            self.dataset = self.data_list[self.n_train_dataset:]

        if self.gpu_id == self.rp_gpu_id:
            print(f'Finished preprocessing {self.mode} dataset...')

        if self.k_shot > self.n_train_dataset:
            if self.k_shot == 12:
                self.n_slice_per_subject = 20 if self.modality == 'fnirs' else 2
            else:
                raise NotImplementedError('please select k_shot==[1,3,6,12]')
        else:
            self.n_slice_per_subject = 10 if self.modality == 'fnirs' else 1

        return self.dataset

    def data_labeing_3c(self, sleep, drowsy, awake):
        data = np.concatenate((sleep, drowsy, awake), axis=0)
        label = [0] * sleep.shape[0] + [1] * drowsy.shape[0] + [2] * awake.shape[0]
        label = np.array(label)
        return data, label

    def data_labeing_2c(self, sleep, awake):
        data = np.concatenate((sleep, awake), axis=0)
        label = [0] * sleep.shape[0] + [1] * awake.shape[0]
        label = np.array(label)
        return data, label

    def is_initial_order(self, order, initial_order):
        return order == initial_order

    def __getitem__(self, index):
        """Return one sample (or set of slices) for drowsiness dataset."""
        sid = self.dataset[index]
        alert_eeg_fname = os.path.join(self.data_dir, sid, 'eeg', 'alertness.npy')
        drowsy_eeg_fname = os.path.join(self.data_dir, sid, 'eeg', 'drowsiness.npy')
        sleep_eeg_fname = os.path.join(self.data_dir, sid, 'eeg', 'sleep.npy')

        alert_fnirs_fname = os.path.join(self.data_dir, sid, 'fnirs', 'alertness.npy')
        drowsy_fnirs_fname = os.path.join(self.data_dir, sid, 'fnirs', 'drowsiness.npy')
        sleep_fnirs_fname = os.path.join(self.data_dir, sid, 'fnirs', 'sleep.npy')

        if self.n_class == 3:
            alert_eeg = np.load(alert_eeg_fname)
            drowsy_eeg = np.load(drowsy_eeg_fname)
            sleep_eeg = np.load(sleep_eeg_fname)

            alert_fnirs = np.load(alert_fnirs_fname)
            drowsy_fnirs = np.load(drowsy_fnirs_fname)
            sleep_fnirs = np.load(sleep_fnirs_fname)

            if self.mode in ('finetune', 'linprobe'):
                alert_eeg = alert_eeg[:self.n_slice_per_subject]
                drowsy_eeg = drowsy_eeg[:self.n_slice_per_subject]
                sleep_eeg = sleep_eeg[:self.n_slice_per_subject]

                alert_fnirs = alert_fnirs[:self.n_slice_per_subject]
                drowsy_fnirs = drowsy_fnirs[:self.n_slice_per_subject]
                sleep_fnirs = sleep_fnirs[:self.n_slice_per_subject]
            elif self.mode == 'test':
                alert_eeg = alert_eeg[:64]
                drowsy_eeg = drowsy_eeg[:64]
                sleep_eeg = sleep_eeg[:64]

                alert_fnirs = alert_fnirs[:64]
                drowsy_fnirs = drowsy_fnirs[:64]
                sleep_fnirs = sleep_fnirs[:64]

            # Pad channels by copying tail channels if needed (keeps original behaviour)
            ncopy_ch = self.target_eeg_shape[0] - alert_eeg.shape[2]
            if ncopy_ch > 0:
                alert_eeg = np.concatenate((alert_eeg, alert_eeg[:, :, -ncopy_ch:, :]), axis=2)
                drowsy_eeg = np.concatenate((drowsy_eeg, drowsy_eeg[:, :, -ncopy_ch:, :]), axis=2)
                sleep_eeg = np.concatenate((sleep_eeg, sleep_eeg[:, :, -ncopy_ch:, :]), axis=2)

            ncopy_ch = self.target_fnirs_shape[0] - alert_fnirs.shape[2]
            if ncopy_ch > 0:
                alert_fnirs = np.concatenate((alert_fnirs, alert_fnirs[:, :, -ncopy_ch:, :]), axis=2)
                drowsy_fnirs = np.concatenate((drowsy_fnirs, drowsy_fnirs[:, :, -ncopy_ch:, :]), axis=2)
                sleep_fnirs = np.concatenate((sleep_fnirs, sleep_fnirs[:, :, -ncopy_ch:, :]), axis=2)

            eeg, label = self.data_labeing_3c(sleep_eeg, drowsy_eeg, alert_eeg)
            fnirs, _ = self.data_labeing_3c(sleep_fnirs, drowsy_fnirs, alert_fnirs)

        elif self.n_class == 2:
            alert_eeg = np.load(alert_eeg_fname)
            sleep_eeg = np.load(sleep_eeg_fname)

            alert_fnirs = np.load(alert_fnirs_fname)
            sleep_fnirs = np.load(sleep_fnirs_fname)

            if self.mode in ('finetune', 'linprobe'):
                alert_eeg = alert_eeg[:self.n_slice_per_subject]
                sleep_eeg = sleep_eeg[:self.n_slice_per_subject]

                alert_fnirs = alert_fnirs[:self.n_slice_per_subject]
                sleep_fnirs = sleep_fnirs[:self.n_slice_per_subject]
            elif self.mode == 'test':
                alert_eeg = alert_eeg[:64]
                sleep_eeg = sleep_eeg[:64]

                alert_fnirs = alert_fnirs[:64]
                sleep_fnirs = sleep_fnirs[:64]

            ncopy_ch = self.target_eeg_shape[0] - alert_eeg.shape[2]
            if ncopy_ch > 0:
                alert_eeg = np.concatenate((alert_eeg, alert_eeg[:, :, -ncopy_ch:, :]), axis=2)
                sleep_eeg = np.concatenate((sleep_eeg, sleep_eeg[:, :, -ncopy_ch:, :]), axis=2)

            ncopy_ch = self.target_fnirs_shape[0] - alert_fnirs.shape[2]
            if ncopy_ch > 0:
                alert_fnirs = np.concatenate((alert_fnirs, alert_fnirs[:, :, -ncopy_ch:, :]), axis=2)
                sleep_fnirs = np.concatenate((sleep_fnirs, sleep_fnirs[:, :, -ncopy_ch:, :]), axis=2)

            eeg, label = self.data_labeing_2c(sleep_eeg, alert_eeg)
            fnirs, _ = self.data_labeing_2c(sleep_fnirs, alert_fnirs)
        else:
            raise NotImplementedError('please select n_class==2 or n_class==3')

        eeg = torch.tensor(eeg, dtype=torch.float32)
        fnirs = torch.tensor(fnirs, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        if self.mode in ('finetune', 'linprobe', 'test'):
            if self.modality == 'eeg':
                return {'x': eeg, 'y': label}
            elif self.modality == 'fnirs':
                return {'x': fnirs, 'y': label}
            elif self.modality == 'multi':
                return {'x1': eeg, 'x2': fnirs, 'y': label}
        else:
            raise NotImplementedError('please select mode in [finetune, linprobe, test]')

    def __len__(self):
        return len(self.dataset)


def seed_worker(worker_id):
    """Worker init for deterministic behavior when using multiple workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(args, mode, batch_size=None):
    """
    Construct a DataLoader for the requested mode.

    Args:
        args: argument namespace (contains workers, multigpu_use, gpu settings, ...)
        mode: one of 'pretrain', 'finetune', 'linprobe', 'test'
        batch_size: optional override for batch size (usually set externally)
    """
    if mode == 'pretrain':
        dataset = Pratrain_data(
            args.target_eeg_size, args.target_fnirs_size,
            args.pretrain_data_dir, mode, batch_size,
            args.gpu_id, args.rp_gpu_id
        )
    elif mode in ('finetune', 'linprobe'):
        modality = args.target_dataset_type.split('_')[-1]
        if args.target_dataset_type == 'sleepstage_eeg':
            dataset = SleepStage(args.target_eeg_size, args.k_shot, args.n_class, args.target_data_dir, mode, args.gpu_id, args.rp_gpu_id)
        elif args.target_dataset_type == 'mental_arithmetic_fnirs':
            dataset = Mental_Arithmetic(args.target_fnirs_size, args.k_shot, args.target_data_dir, mode, args.gpu_id, args.rp_gpu_id)
        elif args.target_dataset_type in ('drowsiness_eeg', 'drowsiness_fnirs', 'drowsiness_multi'):
            dataset = Drowsiness(modality, args.target_eeg_size, args.target_fnirs_size, args.k_shot, args.n_class, args.target_data_dir, mode, args.gpu_id, args.rp_gpu_id)
        else:
            raise NotImplementedError('please select datatype in [sleepstage_eeg, mental_arithmetic_fnirs, drowsiness_eeg, drowsiness_fnirs, drowsiness_multi]')
    elif mode == 'test':
        modality = args.target_dataset_type.split('_')[-1]
        if args.target_dataset_type == 'sleepstage_eeg':
            dataset = SleepStage(args.target_eeg_size, args.k_shot, args.n_class, args.target_data_dir, mode, args.gpu_id, args.rp_gpu_id)
        elif args.target_dataset_type == 'mental_arithmetic_fnirs':
            dataset = Mental_Arithmetic(args.target_fnirs_size, args.k_shot, args.target_data_dir, mode, args.gpu_id, args.rp_gpu_id)
        elif args.target_dataset_type in ('drowsiness_eeg', 'drowsiness_fnirs', 'drowsiness_multi'):
            dataset = Drowsiness(modality, args.target_eeg_size, args.target_fnirs_size, args.k_shot, args.n_class, args.target_data_dir, mode, args.gpu_id, args.rp_gpu_id)
        else:
            raise NotImplementedError('please select datatype in [sleepstage_eeg, mental_arithmetic_fnirs, drowsiness_eeg, drowsiness_fnirs, drowsiness_multi]')
    else:
        raise NotImplementedError('mode must be one of pretrain/finetune/linprobe/test')

    g = torch.Generator()
    g.manual_seed(0)

    # DataLoader construction: keep behaviour consistent with original
    if mode == 'pretrain':
        if args.multigpu_use:
            sampler = DistributedSampler(dataset, shuffle=True, seed=0)
            data_loader = data.DataLoader(
                dataset=dataset, shuffle=False, num_workers=args.workers,
                sampler=sampler, worker_init_fn=seed_worker, generator=g,
            )
        else:
            data_loader = data.DataLoader(
                dataset=dataset, shuffle=(mode != 'test'), num_workers=args.workers,
                worker_init_fn=seed_worker, generator=g,
            )
    elif mode in ('finetune', 'linprobe'):
        if args.multigpu_use:
            sampler = DistributedSampler(dataset, shuffle=True, seed=0)
            data_loader = data.DataLoader(
                dataset=dataset, shuffle=False, num_workers=args.workers,
                batch_size=batch_size, sampler=sampler, worker_init_fn=seed_worker, generator=g,
            )
        else:
            data_loader = data.DataLoader(
                dataset=dataset, shuffle=(mode != 'test'), num_workers=args.workers,
                batch_size=batch_size, worker_init_fn=seed_worker, generator=g,
            )
    elif mode == 'test':
        data_loader = data.DataLoader(
            dataset=dataset, shuffle=(mode != 'test'), num_workers=args.workers,
            worker_init_fn=seed_worker, generator=g,
        )

    return data_loader