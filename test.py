"""
Evaluate a trained model on the validation/test split and print per-class accuracy and a confusion matrix.
Usage:
  python test.py --ckpt logs/modelnet40_ssg/best.pth
  python test.py --ckpt logs/coral_ssg/best.pth --votes 5    # average several random samplings; usually more accurate
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import PointNet2Cls


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default=None, help="defaults to eval_split from the config")
    ap.add_argument("--votes", type=int, default=1, help="how many times to resample and predict each sample before averaging")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    classes = ckpt["classes"]
    ds = build_dataset(cfg, args.split or cfg.get("eval_split", "test"))
    assert ds.classes == classes, "dataset classes do not match the classes used in training"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PointNet2Cls(len(classes), in_channels=ckpt["in_channels"],
                         dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    probs = np.zeros((len(ds), len(classes)), dtype=np.float32)
    for _ in range(args.votes):
        i = 0
        for pts, _lbl in loader:
            p = torch.softmax(model(pts.to(device)), 1).cpu().numpy()
            probs[i:i + len(p)] += p
            i += len(p)
    pred = probs.argmax(1)
    labels = ds.labels

    conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(labels, pred):
        conf[t, p] += 1
    per_cls = np.diag(conf) / np.maximum(conf.sum(1), 1)
    print(f"\nSamples {len(ds)} | overall accuracy OA = {np.trace(conf) / conf.sum():.4f} | "
          f"mean class accuracy mAcc = {per_cls[conf.sum(1) > 0].mean():.4f}\n")
    print("Per-class accuracy:")
    for c, a, n in zip(classes, per_cls, conf.sum(1)):
        print(f"  {c:<20s} {a:.3f}  (n={n})")
    if len(classes) <= 15:
        print("\nConfusion matrix (rows = true, columns = predicted):")
        w = max(len(c) for c in classes)
        print(" " * (w + 2) + " ".join(f"{c[:6]:>6s}" for c in classes))
        for c, row in zip(classes, conf):
            print(f"{c:>{w}s}  " + " ".join(f"{v:6d}" for v in row))


if __name__ == "__main__":
    main()
