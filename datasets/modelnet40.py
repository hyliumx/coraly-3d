"""
ModelNet40 dataset (modelnet40_normal_resampled version: 10000 points per shape, with normals).

Directory layout:
  data/modelnet40_normal_resampled/
    modelnet40_shape_names.txt
    modelnet40_train.txt
    modelnet40_test.txt
    airplane/airplane_0001.txt   # each line: x,y,z,nx,ny,nz
    ...

The first run caches the first num_points points of every shape into one .npz; later loads are fast.
"""
import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .transforms import augment, normalize_unit_sphere, sample_points


class ModelNet40(Dataset):
    def __init__(self, root, split="train", num_points=1024, use_normals=False, train_aug=True, **_):
        self.root, self.split, self.num_points = root, split, num_points
        self.use_normals = use_normals
        self.do_aug = train_aug and split == "train"
        with open(os.path.join(root, "modelnet40_shape_names.txt")) as f:
            self.classes = [l.strip() for l in f if l.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        cache = os.path.join(root, f"cache_{split}_{num_points}pts.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            self.points, self.labels = d["points"], d["labels"]
        else:
            self.points, self.labels = self._build_cache(split, cache)

    def _build_cache(self, split, cache):
        with open(os.path.join(self.root, f"modelnet40_{split}.txt")) as f:
            ids = [l.strip() for l in f if l.strip()]
        pts_all, labels = [], []
        for shape_id in tqdm(ids, desc=f"Reading ModelNet40 {split} for the first time (cached afterward)"):
            cls = "_".join(shape_id.split("_")[:-1])
            path = os.path.join(self.root, cls, shape_id + ".txt")
            pts = np.loadtxt(path, delimiter=",", dtype=np.float32, max_rows=self.num_points)
            pts_all.append(sample_points(pts, self.num_points, "first"))
            labels.append(self.class_to_idx[cls])
        pts_all, labels = np.stack(pts_all), np.array(labels, dtype=np.int64)
        np.savez(cache, points=pts_all, labels=labels)
        return pts_all, labels

    @property
    def in_channels(self):
        return 3 if self.use_normals else 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        pts = self.points[i].copy()
        pts[:, :3] = normalize_unit_sphere(pts[:, :3])
        if not self.use_normals:
            pts = pts[:, :3]
        if self.do_aug:
            # ModelNet objects are all upright and the original implementation does not rotate; only scale/shift here
            pts = augment(pts, rotate_z=False, jitter=0)
        return pts.T.astype(np.float32), self.labels[i]   # [C, N]
