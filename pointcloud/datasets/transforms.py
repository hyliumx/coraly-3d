"""Point cloud preprocessing and augmentation. Inputs are numpy arrays [N, 3+C]; the first 3 columns are xyz."""
import numpy as np


def normalize_unit_sphere(xyz):
    """Shift to the centroid, then scale into the unit sphere so objects/corals of different sizes share one scale."""
    xyz = xyz - xyz.mean(axis=0)
    scale = np.max(np.linalg.norm(xyz, axis=1))
    return xyz / (scale + 1e-8)


def farthest_point_sample_np(points, npoint):
    """Numpy farthest point sampling. Covers the cloud more evenly than random sampling, but is slower."""
    N = points.shape[0]
    xyz = points[:, :3]
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.full(N, 1e10)
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        dist = np.sum((xyz - xyz[farthest]) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)
    return points[centroids]


def sample_points(points, npoint, method="random"):
    """Turn a point cloud of any size into exactly npoint points. If there are too few, resample with replacement."""
    N = points.shape[0]
    if N < npoint:
        idx = np.concatenate([np.arange(N), np.random.choice(N, npoint - N, replace=True)])
        return points[idx]
    if method == "fps":
        return farthest_point_sample_np(points, npoint)
    if method == "first":           # preprocessed ModelNet data can just take the first N points
        return points[:npoint]
    return points[np.random.choice(N, npoint, replace=False)]


def augment(points, rotate_z=True, scale=(0.8, 1.25), shift=0.1, jitter=0.01):
    """Random training augmentation: rotation about the vertical (z) axis, global scaling, translation, per-point jitter.
    Only xyz is changed; color and other features stay the same. (Note: if the extra features are normals,
    they are not rotated along with the points.)"""
    pts = points.copy()
    xyz = pts[:, :3]
    if rotate_z:
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ R.T
    if scale:
        xyz = xyz * np.random.uniform(*scale)
    if shift:
        xyz = xyz + np.random.uniform(-shift, shift, size=(1, 3))
    if jitter:
        xyz = xyz + np.clip(jitter * np.random.randn(*xyz.shape), -5 * jitter, 5 * jitter)
    pts[:, :3] = xyz
    return pts.astype(np.float32)
