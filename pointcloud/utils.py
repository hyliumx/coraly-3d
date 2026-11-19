import random

import numpy as np
import torch
import yaml


def load_config(path, overrides=()):
    """Read a YAML config. overrides look like ["train.epochs=5", "data.num_points=512"]."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, val = ov.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    # Give each DataLoader worker a different numpy seed (otherwise augmentation repeats)
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    """Return overall accuracy, mean class accuracy, and confusion matrix (rows = true, columns = predicted)."""
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pts, labels in loader:
        pts = pts.to(device)
        pred = model(pts).argmax(1).cpu().numpy()
        for t, p in zip(labels.numpy(), pred):
            conf[t, p] += 1
    oa = np.trace(conf) / max(conf.sum(), 1)
    per_cls = np.diag(conf) / np.maximum(conf.sum(1), 1)
    macc = per_cls[conf.sum(1) > 0].mean() if (conf.sum(1) > 0).any() else 0.0
    return oa, macc, conf
