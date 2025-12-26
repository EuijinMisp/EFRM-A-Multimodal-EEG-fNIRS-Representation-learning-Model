# EFRM: A Multimodal EEG–fNIRS Representation Learning Model

This repository implements EFRM — an EEG & fNIRS multimodal foundation model using Masked Autoencoders (MAE) combined with a CLIP-like contrastive objective. It includes pretraining code for the multimodal MAE, downstream fine‑tuning and linear‑probing scripts, data loaders, utilities, and simple WandB logging integration.

This README gives concrete, focused instructions to prepare data and run the included main scripts (pretrain / finetune / linearprobe), explains the code layout, and includes the paper abstract and reference.

---

Table of contents
- Abstract
- Reference
- Overview
- Repository layout
- Data format and expected layout
- How to run (concrete commands)
  - Pretraining (main_pretrain.py)
  - Fine‑tuning (main_finetune.py)
  - Linear‑probing (main_linearprobe.py)
  - Testing
- Important CLI arguments
- Multi‑GPU / distributed notes
- Checkpoints & logging
- Short code walkthrough
- Next steps and helpful tips
- Contact

Abstract
--------
Recent advances in brain signal analysis highlight the need for robust classifiers that can be trained with minimal labeled data. To meet this demand, transfer learning has emerged as a promising strategy: large-scale unlabeled data is used to train pre-trained models, which are later adapted with minimal labeled data. However, while most existing transfer learning studies focus primarily on electroencephalography (EEG) signals, their generalization to other brain signal modalities such as functional near-infrared spectroscopy (fNIRS) remains limited. To address this issue, we propose a multimodal representation model compatible with EEG-only, fNIRS-only, and paired EEG–fNIRS datasets. The proposed method consists of two stages: a pre-training stage that learns both modality-specific and shared representations across EEG and fNIRS, followed by a transfer learning stage adapted to specific downstream tasks. By leveraging the shared domain across EEG and fNIRS, our model outperforms single-modality approaches. We constructed pre-training datasets containing approximately 1250 h of brain signal recordings from 918 participants. Unlike previous multimodal approaches that require both EEG and fNIRS data for training, our method enables adaptation to single-modality datasets, enhancing flexibility and practicality. Experimental results demonstrate that our method achieves competitive performance in comparison with state-of-the-art supervised learning models, even with minimal labeled data. Our method also outperforms previously pre-trained models, showing especially significant improvements in fNIRS classification performance.

Keywords: EEG; fNIRS; Multimodal representation learning; Transfer learning; Few-shot learning

Reference
---------
Euijin Jung, Jinung An, "EFRM: A Multimodal EEG–fNIRS Representation-learning Model for few-shot brain-signal classification", Computers in Biology and Medicine, Volume 199, 2025, 111292.  
DOI: https://doi.org/10.1016/j.compbiomed.2025.111292  
ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0010482525016464

Suggested citation (BibTeX)
```bibtex
@article{jung2025efrm,
  title={EFRM: A Multimodal EEG–fNIRS Representation-learning Model for few-shot brain-signal classification},
  author={Jung, Euijin and An, Jinung},
  journal={Computers in Biology and Medicine},
  volume={199},
  pages={111292},
  year={2025},
  doi={10.1016/j.compbiomed.2025.111292}
}
```

Overview
--------
EFRM pretrains two MAE encoders (one for EEG, one for fNIRS) optimizing reconstruction losses plus a modality alignment loss computed from pooled encoder features. After pretraining, encoder weights can be loaded into classification wrappers for downstream tasks (sleep staging, mental arithmetic, drowsiness detection). The code supports single‑GPU and multi‑GPU (distributed) execution and integrates with Weights & Biases (WandB) for logging (optional).

Repository layout
-----------------
- code/
  - main_pretrain.py        — pretraining entrypoint
  - main_finetune.py        — fine‑tuning entrypoint
  - main_linearprobe.py     — linear‑probe entrypoint (train head only)
  - dataloader.py           — dataset classes and get_loader()
  - model_pretrain.py       — MAE implementation and multimodal wrapper
  - model_finetune.py       — classification wrappers for downstream tasks
  - learn.py                — solver (training/eval loops, checkpointing)
  - utils.py                — augmentations, initialization, helper functions
  - log.py                  — WandB helper
- configs/                  — optional example configs
- data/                     — user-supplied datasets (not included)
- README.md                 — this file

Data format and expected layout
------------------------------
Data is not included. Prepare your data locally according to the shapes and folder structure expected by the dataloader.

1) Pretraining data (default base dir: ../data/pretrain)
- EEG-only:
  - ../data/pretrain/EEG_only/<collection_or_subject>/*.npy
  - Each EEG .npy: numpy array of shape [channels, time] (e.g. [24, 1024])
- fNIRS-only:
  - ../data/pretrain/fNIRS_only/<collection_or_subject>/*.npy
  - Each fNIRS .npy: numpy array of shape [2, channels, time]
- Paired EEG–fNIRS:
  - ../data/pretrain/EEG-fNIRS/<collection_or_subject>/*/
    - must include:
      - eeg.npy  → expected shape [B, 1, nch, ntime]
      - fnirs.npy → expected shape [B, 2, nch, ntime]

2) Downstream tasks (examples)
- SleepStage (EEG):
  - ../data/downstream/EEG/sleepstage/<subject>/
    - awake.npy, nonrem.npy, rem.npy (each with slice dimension)
- Mental arithmetic (fNIRS):
  - ../data/downstream/fNIRS/mental_arithmetic/<subject>/
    - bl.npy, ma.npy
- Drowsiness (multi-modal):
  - ../data/downstream/EEG-fNIRS/drowsiness/<subject>/
    - eeg/alertness.npy, eeg/drowsiness.npy, eeg/sleep.npy
    - fnirs/alertness.npy, fnirs/drowsiness.npy, fnirs/sleep.npy

Notes:
- The dataloader performs cropping, channel expansion (repeat or mirror), and basic augmentations automatically. See code/dataloader.py for exact behavior.

How to run (concrete commands)
------------------------------

Important: when passing `--gpu_ids` from a shell, pass the Python list as a quoted string, e.g. `--gpu_ids "[0]"` or `--gpu_ids "[0,1]"`.

1) Pretraining — code/main_pretrain.py
Single‑GPU example:
```bash
python code/main_pretrain.py --gpu_ids "[0]" --batch_size 32 --pre_niter 300 --pre_lr 0.0001 --run_name "EFRM_pretrain_v1"
```

Multi‑GPU example:
```bash
python code/main_pretrain.py --gpu_ids "[0,1]" --batch_size 64 --pre_niter 300 --pre_lr 0.0001 --run_name "EFRM_pretrain_v1"
```

Key pretrain args:
- --batch_size: global batch size (divided by world_size internally)
- --pre_niter: number of iterations/epochs
- --pre_lr: learning rate
- --mask_ratio: MAE mask ratio
- --pre_model_save_path: checkpoint directory

2) Fine‑tuning — code/main_finetune.py
Example:
```bash
python code/main_finetune.py --gpu_ids "[0]" --mode finetune \
  --target_dataset_type sleepstage_eeg --k_shot 1 --n_class 2 \
  --batch_size 16 --fine_niter 50 --fine_lr 0.0001 \
  --pretrained_model_path "../run/pretrain" --run_name "EFRM_finetune_v1"
```

3) Linear‑probing — code/main_linearprobe.py
Example:
```bash
python code/main_linearprobe.py --gpu_ids "[0]" --mode linprobe \
  --target_dataset_type sleepstage_eeg --k_shot 1 --n_class 2 \
  --batch_size 16 --fine_niter 200 --fine_lr 0.0001 \
  --pretrained_model_path "../run/pretrain" --run_name "EFRM_linprobe_v1"
```

4) Testing / inference
- Use `mode='test'` and provide the appropriate checkpoint paths and test data folders. See learn.py for solver.test and prediction details.

Important CLI arguments (summary)
--------------------------------
Common:
- --mode: pretrain | finetune | linprobe | test
- --gpu_ids: quoted Python list of GPU ids, e.g. `"[0]"` or `"[0,1]"`
- --run_name: name used for saving and logging
- --model_name: default 'EFMF' retained for compatibility
- --batch_size: global batch size (the scripts divide across GPUs)
- --workers: number of data loader workers
- --seed: RNG seed

Pretrain:
- --pre_model_save_path, --pre_lr, --pre_niter, --mask_ratio

Finetune/linprobe:
- --pretrained_model_path, --fine_model_save_path, --fine_lr, --fine_niter, --target_dataset_type, --k_shot, --n_class

Multi‑GPU / distributed notes
-----------------------------
- Scripts spawn processes via torch.multiprocessing when multiple GPU IDs are provided.
- Distributed init uses NCCL and a TCP port (default 12355). Ensure the GPUs and environment support distributed training and the port is available.

Checkpoints & logging
---------------------
- Checkpoints saved under `{save_path}/{run_name}/last_model` and `best_model` (finetune).
- WandB logging is used (optional). If you don't use WandB, disable or stub `log.init_wandb` calls.

Short code walkthrough
----------------------
- code/dataloader.py: dataset_option, Pratrain_data, downstream datasets, get_loader
- code/model_pretrain.py: PatchEmbed, MaskedAutoencoderViT, MAE multimodal wrapper
- code/model_finetune.py: classifier wrappers (EEG / fNIRS / EF multimodal)
- code/learn.py: solver class — setup, training loops, checkpoint load/save, evaluation
- code/log.py: WandB helper
- code/utils.py: augmentations, initialization, pos-embed interpolation, measurement helpers

Next steps and helpful tips
--------------------------
- Create small synthetic numpy arrays matching expected shapes to sanity-check the pipeline.
- If you do not plan to use WandB, remove or stub logging calls.
- Start single‑GPU runs to confirm everything works before scaling to multi‑GPU.

Contact & contributing
----------------------
- Contributions welcome: open an issue for bugs or feature requests, then submit a PR.
- Add a LICENSE file at the repository root before publicly publishing.

