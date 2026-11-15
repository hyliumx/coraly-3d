"""
Coral point cloud dataset (placeholder implementation; adjust once the real data format is known).

Assumed directory layout (one folder per growth form; the folder name is the class name):
  data/coral/
    branching/
      colony_001.ply
      colony_002.las
    massive/
      ...
    encrusting/
      ...
    splits/            # optional fixed split. If missing, a stratified random split using val_ratio is made
      train.txt        # one relative path per line, e.g. branching/colony_001.ply
      val.txt

Supported formats: .ply (plyfile), .las/.laz (laspy), .npy ([N,3] or [N,6]), .txt/.xyz (space- or comma-separated)
Each point reads xyz and optionally rgb (normalized to 0-1).

SfM reconstructions often have hundreds of thousands to millions of points, so:
  1. After loading, randomly downsample to preload_points and keep them in memory (when cache_in_memory=True)
  2. Each time a sample is fetched, sample num_points from that (random or fps)
"""
import os
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset

from .transforms import augment, normalize_unit_sphere, sample_points

EXTS = (".ply", ".las", ".laz", ".npy", ".txt", ".xyz")


def _rgb_to_unit(rgb):
    rgb = rgb.astype(np.float32)
    maxv = rgb.max() if rgb.size else 1.0
    if maxv > 255:      # colors in LAS files are usually 16-bit (0-65535)
        rgb /= 65535.0
    elif maxv > 1:
        rgb /= 255.0
    return rgb


def load_point_cloud(path, use_rgb=True):
    """Read one point cloud file. Returns a float32 array of shape [N, 3] or [N, 6] (xyz + rgb)."""
    ext = os.path.splitext(path)[1].lower()
    rgb = None
    if ext == ".ply":
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
        names = v.data.dtype.names
        if use_rgb and all(c in names for c in ("red", "green", "blue")):
            rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1)
    elif ext in (".las", ".laz"):
        import laspy
        las = laspy.read(path)
        xyz = np.stack([las.x, las.y, las.z], axis=1)
        if use_rgb and hasattr(las, "red"):
            rgb = np.stack([las.red, las.green, las.blue], axis=1)
    elif ext == ".npy":
        arr = np.load(path)
        xyz = arr[:, :3]
        if use_rgb and arr.shape[1] >= 6:
            rgb = arr[:, 3:6]
    else:  # .txt / .xyz
        with open(path) as f:
            first = f.readline()
        arr = np.loadtxt(path, delimiter="," if "," in first else None)
        xyz = arr[:, :3]
        if use_rgb and arr.shape[1] >= 6:
            rgb = arr[:, 3:6]

    xyz = np.asarray(xyz, dtype=np.float32)
    if use_rgb:
        if rgb is None:
            print(f"[warning] {path} has no color information; using 0.5 gray instead")
            rgb = np.full_like(xyz, 0.5)
        return np.concatenate([xyz, _rgb_to_unit(np.asarray(rgb))], axis=1)
    return xyz


def scan_files(root, classes=None):
    """Scan all point cloud files under root/<class name>/. Returns (list of relative paths, list of class names)."""
    if classes is None:
        classes = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)) and d != "splits")
    files = []
    for c in classes:
        for fn in sorted(os.listdir(os.path.join(root, c))):
            if fn.lower().endswith(EXTS):
                files.append(f"{c}/{fn}")
    return files, classes


def stratified_split(files, val_ratio, seed):
    """Stratified random split by class, so every class has samples in the validation set."""
    by_cls = defaultdict(list)
    for f in files:
        by_cls[f.split("/")[0]].append(f)
    rng = random.Random(seed)
    train, val = [], []
    for c, fs in sorted(by_cls.items()):
        fs = fs[:]
        rng.shuffle(fs)
        n_val = max(1, int(round(len(fs) * val_ratio))) if len(fs) > 1 else 0
        val += fs[:n_val]
        train += fs[n_val:]
    return train, val


class CoralDataset(Dataset):
    def __init__(self, root, split="train", num_points=4096, use_rgb=True, classes=None,
                 val_ratio=0.2, seed=42, sampling="random", preload_points=20000,
                 cache_in_memory=True, train_aug=True, **_):
        self.root, self.split, self.num_points = root, split, num_points
        self.use_rgb, self.sampling, self.preload_points = use_rgb, sampling, preload_points
        self.do_aug = train_aug and split == "train"

        all_files, self.classes = scan_files(root, classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        split_file = os.path.join(root, "splits", f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.files = [l.strip() for l in f if l.strip()]
        else:
            train, val = stratified_split(all_files, val_ratio, seed)
            self.files = train if split == "train" else val
        self.labels = np.array([self.class_to_idx[f.split("/")[0]] for f in self.files], dtype=np.int64)

        self._cache = {}
        if cache_in_memory:
            for i in range(len(self.files)):
                self._cache[i] = self._load(i)

    def _load(self, i):
        pts = load_point_cloud(os.path.join(self.root, self.files[i]), self.use_rgb)
        if self.preload_points and pts.shape[0] > self.preload_points:
            pts = pts[np.random.choice(pts.shape[0], self.preload_points, replace=False)]
        return pts

    @property
    def in_channels(self):
        return 3 if self.use_rgb else 0

    def class_counts(self):
        return np.bincount(self.labels, minlength=len(self.classes))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        pts = self._cache[i] if i in self._cache else self._load(i)
        pts = sample_points(pts, self.num_points, self.sampling).copy()
        pts[:, :3] = normalize_unit_sphere(pts[:, :3])
        if self.do_aug:
            # Corals in the wild have no fixed "front", so random rotation about the vertical axis is reasonable
            pts = augment(pts, rotate_z=True)
        return pts.T.astype(np.float32), self.labels[i]
