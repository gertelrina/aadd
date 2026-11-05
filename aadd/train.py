"""Training loop for per-chunk degradation classifiers on featurized ``.npz`` data.

Example::

    aadd featurize data/train_dev_clean --out feats/train_mel.npz --features mel
    aadd featurize data/test_clean     --out feats/test_mel.npz  --features mel
    aadd train --train feats/train_mel.npz --test feats/test_mel.npz --out weights/aadd_v2_efficientnet_b0_mel.pt

Model selection uses a speaker-disjoint validation split carved out of the
training set (``--val-fraction``); the ``--test`` set is only reported.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aadd.data.dataset import ChunkArrayDataset, SpecMask, split_by_speaker
from aadd.metrics import compute_metrics
from aadd.model import build_model, pick_device, save_checkpoint


@torch.no_grad()
def predict_dataset(
    model: nn.Module, ds: ChunkArrayDataset, device, batch_size: int = 256, workers: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    probs, ys = [], []
    for x, y in dl:
        probs.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train(
    train_npz: str,
    out_path: str,
    test_npz: str | None = None,
    arch: str = "efficientnet_b0",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    val_fraction: float = 0.1,
    seed: int = 0,
    device: str | None = None,
    workers: int = 0,
    pretrained: bool = True,
    spec_mask: bool = True,
    select_metric: str = "f1_macro",
    name: str = "v2",
    log_path: str | None = None,
    limit: int | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = pick_device(device)

    full = ChunkArrayDataset(train_npz)
    tr_idx, va_idx = split_by_speaker(full, val_fraction=val_fraction, seed=seed)
    if limit:
        tr_idx = tr_idx[:limit]
        va_idx = va_idx[: max(1, limit // 10)]
    aug = SpecMask() if (spec_mask and full.spec.kind == "mel") else None
    train_ds = ChunkArrayDataset(train_npz, tr_idx, augment=aug)
    val_ds = ChunkArrayDataset(train_npz, va_idx)
    test_ds = ChunkArrayDataset(test_npz) if test_npz else None
    in_channels = int(full.X.shape[1])
    classes = full.classes
    print(
        f"device={dev} train={len(train_ds)} val={len(val_ds)} (speakers disjoint) test={len(test_ds) if test_ds else 0} "
        f"input={tuple(full.X.shape[1:])} classes={len(classes)} arch={arch}",
        flush=True,
    )

    model = build_model(arch, num_classes=len(classes), in_channels=in_channels, pretrained=pretrained).to(dev)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps), pct_start=0.15)
    crit = nn.BCEWithLogitsLoss()

    history, best, best_epoch = [], -1.0, -1
    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        t0 = time.time()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
        yv, pv = predict_dataset(model, val_ds, dev, workers=workers)
        mv = compute_metrics(yv, pv, classes)
        rec = {
            "epoch": epoch,
            "train_loss": tot / max(n, 1),
            "val_f1_macro": mv["f1_macro"],
            "val_f1_micro": mv["f1_micro"],
            "val_exact_match": mv["exact_match"],
            "val_per_class_f1": {c: mv["per_class"][c]["f1"] for c in classes},
            "lr": sched.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        if test_ds is not None:
            yt, pt = predict_dataset(model, test_ds, dev, workers=workers)
            mt = compute_metrics(yt, pt, classes)
            rec["test_f1_macro"] = mt["f1_macro"]
            rec["test_f1_micro"] = mt["f1_micro"]
            rec["test_per_class_f1"] = {c: mt["per_class"][c]["f1"] for c in classes}
        history.append(rec)
        score = mv[select_metric]
        marker = ""
        if score > best:
            best, best_epoch = score, epoch
            marker = " *"
            save_checkpoint(
                out_path,
                model,
                arch,
                in_channels,
                full.spec,
                classes=classes,
                meta={
                    "name": name,
                    "epoch": epoch,
                    "trained_on": str(train_npz),
                    "train_chunks": len(train_ds),
                    "val_metrics": mv,
                    "test_metrics": mt if test_ds is not None else None,
                    "config": {
                        "arch": arch,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "val_fraction": val_fraction,
                        "seed": seed,
                        "pretrained": pretrained,
                        "spec_mask": spec_mask,
                        "select_metric": select_metric,
                    },
                },
            )
        print(
            f"epoch {epoch:02d} loss {rec['train_loss']:.4f} val macroF1 {mv['f1_macro']:.4f} microF1 {mv['f1_micro']:.4f} "
            + (f"test macroF1 {rec['test_f1_macro']:.4f} " if test_ds is not None else "")
            + f"({rec['seconds']:.0f}s){marker}",
            flush=True,
        )
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump({"history": history, "best_epoch": best_epoch, "best": best}, f, indent=1)
    return {
        "best_epoch": best_epoch,
        "best": best,
        "history": history,
        "out": str(out_path),
        "seconds": time.time() - t_start,
    }
