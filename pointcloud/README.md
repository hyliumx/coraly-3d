# Coraly-3D: Classifying Coral Growth Forms Directly from 3D Point Clouds with PointNet++

This is a **baseline code framework**. First get it running on the public ModelNet40 dataset to confirm the environment and code work.
Then put your coral point clouds (for example SfM reconstructions with RGB from the 100 Island Challenge) in the expected layout and switch to the coral config to train.

The model is adapted from [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) (MIT License).
Original paper: Qi et al., *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*, NeurIPS 2017.
It is pure PyTorch, with no CUDA extensions to compile, so it also works on Windows.

## Project layout

```
coraly-3d/
├── configs/
│   ├── modelnet40.yaml      # ModelNet40 config (step 1)
│   └── coral.yaml           # Coral data config (step 2)
├── datasets/
│   ├── modelnet40.py        # ModelNet40 loader
│   ├── coral.py             # Coral data loader
│   └── transforms.py        # Normalization, sampling, augmentation
├── models/
│   ├── pointnet2_utils.py   # FPS, ball query, Set Abstraction layer
│   └── pointnet2_cls.py     # PointNet++ (SSG) classification network
├── train.py                 # Training
├── test.py                  # Evaluation (per-class accuracy + confusion matrix)
└── utils.py
```

## 1. Set up the environment

Using conda is recommended (Python 3.10-3.12 all work). If you don't have conda, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) first,
then run `conda init powershell` once in PowerShell and **reopen the terminal** before continuing:

```bash
conda create -n coraly -c conda-forge --override-channels python=3.11 -y
conda activate coraly
```

> `-c conda-forge --override-channels` avoids Anaconda's default channels, so you don't have to accept their terms of service first.

Install PyTorch (GPU build). An RTX 4060 works with a recent CUDA version. **For the most accurate command, pick your system at https://pytorch.org/get-started/locally/**. For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then install the other dependencies:

```bash
pip install -r requirements.txt
```

Check that the GPU is available (should print `True`):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Train on ModelNet40

1. Download `modelnet40_normal_resampled.zip` (about 1.6GB). The original URL
   `https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip` often doesn't open,
   so search Kaggle or Hugging Face for `modelnet40_normal_resampled` and download the file with the same name.
   Make sure you get the **point cloud version (each shape is a .txt file, each line `x,y,z,nx,ny,nz`)**, not the original Princeton version (`.off` meshes), which the loader cannot read.
2. Extract it into `data/` at the project root (create it if missing; note it is `data/`, not the code folder `datasets/`). The result should be
   `data/modelnet40_normal_resampled/airplane/...`, and that folder should directly contain
   `modelnet40_shape_names.txt`, `modelnet40_train.txt`, and `modelnet40_test.txt` (9843 training and 2468 test shapes).
   Watch out for an extra nested folder with the same name after extraction.
3. Train:

```bash
python train.py --config configs/modelnet40.yaml
```

The first run spends a few minutes reading the txt files and building a cache (`cache_*.npz`); later starts are fast.
Reference result: PointNet++ SSG reaches about **92%** overall accuracy on ModelNet40 (yanx27 reports 92.2%, without normals).
For a quick look, add `--set train.epochs=10`. If data loading errors on Windows, also add `train.num_workers=0`.

4. Evaluate:

```bash
python test.py --ckpt logs/modelnet40_ssg/best.pth --votes 3
```

## 3. Switch to coral data

Arrange the data with "one folder per growth form":

```
data/coral/
├── branching/      colony_001.ply  colony_002.las ...
├── massive/        ...
├── encrusting/     ...
└── splits/         # optional: train.txt / val.txt, one relative path per line
```

- Folder names are the class names, and the number of classes is detected automatically, so **no code changes are needed**.
- Without `splits/`, a stratified random 20% per class is held out for validation.
  If the same coral has multiple scans, or you want to split by island/plot (to avoid data leakage), write the `splits/` files yourself.
- Each point reads xyz + rgb; colors are normalized to 0-1 automatically (8-bit and 16-bit colors are both handled).

Then:

```bash
python train.py --config configs/coral.yaml
python test.py --ckpt logs/coral_ssg/best.pth --votes 5
```

## 4. Common config options

All parameters live in the yaml files. You can also override them temporarily with `--set key=value` (without editing the file):

| Option | Meaning | Suggestion |
|---|---|---|
| `data.num_points` | Points sampled per sample | 1024 for ModelNet; 2048-4096 for coral |
| `data.use_rgb` | Whether to use color as a feature | Worth a comparison experiment: with/without color |
| `data.sampling` | `random` or `fps` | `fps` covers more evenly but is slower |
| `train.batch_size` | Batch size | Lower it if you hit CUDA out of memory |
| `train.class_weighted_loss` | Weight the loss by class frequency | Turn on when classes are imbalanced |
| `train.num_workers` | Number of data loading processes | Set to 0 if it errors on Windows |

Example: `python train.py --config configs/coral.yaml --set data.num_points=2048 train.batch_size=16`

## 5. Outputs

Training results go to `logs/<exp_name>/`:
- `best.pth`: the model with the highest validation accuracy
- `last.pth`: the model from the last epoch
- `log.txt`: loss and accuracy for each epoch
- `config.yaml`: the full config used for this run (for reproducibility)

## 6. What may need adjusting once you have real coral data

- **File format**: if it isn't .ply/.las/.npy/.txt, add a branch in `load_point_cloud()` in `datasets/coral.py`.
- **Several corals in one file** (a whole plot): cut each colony out using the annotations first, then place them in the layout above.
- **Scale**: every sample is currently normalized to the unit sphere, so the model cannot see a coral's absolute size. If size helps distinguish growth forms, add the original scale as an extra feature.
- **Small datasets**: you may only have a few hundred coral samples. With so few samples BatchNorm makes validation accuracy fluctuate, so train for more epochs and evaluate with `--votes`.
