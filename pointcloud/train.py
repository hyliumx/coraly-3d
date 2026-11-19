"""
Train a PointNet++ classifier.
Usage:
  python train.py --config configs/modelnet40.yaml
  python train.py --config configs/coral.yaml
  python train.py --config configs/modelnet40.yaml --set train.epochs=5   # override config values
Outputs go to logs/<exp_name>/: best.pth (best validation model), last.pth, log.txt, config.yaml
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import PointNet2Cls
from utils import evaluate, load_config, set_seed, worker_init_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config values, e.g. train.epochs=5")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tc = cfg["train"]
    set_seed(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    out_dir = os.path.join("logs", cfg["exp_name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    train_set = build_dataset(cfg, "train")
    val_set = build_dataset(cfg, cfg.get("eval_split", "test"))
    classes = train_set.classes
    log(f"Classes {len(classes)} | train {len(train_set)} | val {len(val_set)} | {cfg['data']['num_points']} points per sample")

    loader_kw = dict(num_workers=tc["num_workers"], worker_init_fn=worker_init_fn,
                     pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=tc["batch_size"], shuffle=True,
                              drop_last=len(train_set) > tc["batch_size"], **loader_kw)
    val_loader = DataLoader(val_set, batch_size=tc["batch_size"], shuffle=False, **loader_kw)

    model = PointNet2Cls(len(classes), in_channels=train_set.in_channels,
                         dropout=cfg["model"]["dropout"]).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    weight = None
    if tc.get("class_weighted_loss"):
        counts = np.bincount(train_set.labels, minlength=len(classes)).astype(np.float32)
        w = counts.sum() / (len(classes) * np.maximum(counts, 1))
        weight = torch.tensor(w, device=device)
        log("Samples per class: " + ", ".join(f"{c}={int(n)}" for c, n in zip(classes, counts)))
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=tc.get("label_smoothing", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, tc["lr_step"], tc["lr_gamma"])
    use_amp = tc.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_oa = -1.0
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, total_loss, correct, seen = time.time(), 0.0, 0, 0
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(pts)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        scheduler.step()

        oa, macc, _ = evaluate(model, val_loader, len(classes), device)
        is_best = oa > best_oa
        ckpt = {"model": model.state_dict(), "epoch": epoch, "classes": classes,
                "in_channels": train_set.in_channels, "cfg": cfg, "val_oa": oa}
        torch.save(ckpt, os.path.join(out_dir, "last.pth"))
        if is_best:
            best_oa = oa
            torch.save(ckpt, os.path.join(out_dir, "best.pth"))
        log(f"epoch {epoch:3d} | loss {total_loss / max(seen, 1):.4f} | train acc {correct / max(seen, 1):.4f} | "
            f"val OA {oa:.4f} mAcc {macc:.4f}{' *best' if is_best else ''} | {time.time() - t0:.0f}s")
    log(f"Training finished. Best val OA = {best_oa:.4f}, model saved to {out_dir}/best.pth")


if __name__ == "__main__":
    main()
