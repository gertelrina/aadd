"""Train a v3 frame-level model on on-the-fly synthetic data.

python -m aadd.v3.train --pools POOLS --val VAL_EN.npz VAL_RU.npz --arch whisper-small --out OUT.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aadd.v3.data import FixedSet, SynthDataset
from aadd.v3.metrics import evaluate_arrays
from aadd.v3.model import build, masked_bce
from aadd.v3.synth import CLASSES_V3

FORMAT = "aadd-v3-checkpoint-1"


@torch.no_grad()
def predict(model, ds, device, bs: int = 64, workers: int = 8) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in DataLoader(ds, batch_size=bs, num_workers=workers):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(x.to(device, non_blocking=True)).float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def evaluate_sets(model, paths, device) -> dict:
    res = {}
    for p in paths:
        ds = FixedSet(p)
        P = predict(model, ds, device)
        res[Path(p).stem] = evaluate_arrays(ds.Y.astype(np.float32), P, ds.IG, CLASSES_V3)
    return res


def save(path, model, arch, meta):
    sd = {
        k: v.detach().to(torch.float16 if v.is_floating_point() else v.dtype).cpu()
        for k, v in model.state_dict().items()
    }
    torch.save(
        {
            "format": FORMAT,
            "arch": arch,
            "classes": CLASSES_V3,
            "frame_seconds": 0.02,
            "window_seconds": 8.0,
            "state_dict": sd,
            "meta": meta,
        },
        path,
    )


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build(ck["arch"], **({"pretrained": False} if ck["arch"].startswith("whisper") else {}))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return model.to(device).eval(), ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--arch", default="whisper-small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5, help="backbone lr (whisper)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", default=None, help="v3 checkpoint to distil from (soft targets mixed 50/50)")
    ap.add_argument("--log", default=None)
    ap.add_argument("--init", default=None, help="start from this v3 checkpoint (fine-tuning)")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build(a.arch).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
        print(f"initialised from {a.init}", flush=True)
    backbone = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": head, "lr": a.lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": a.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        f = (
            step / a.warmup
            if step < a.warmup
            else 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))
        )
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f

    teacher = None
    if a.teacher:
        teacher, _ = load(a.teacher, dev)
        teacher.requires_grad_(False)

    ds = SynthDataset(a.pools, "train", seed=a.seed, length=a.steps * a.bs)
    dl = DataLoader(
        ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"arch={a.arch} params={n_params:.1f}M device={dev} steps={a.steps} bs={a.bs} workers={a.workers}", flush=True
    )

    hist, best, t0, tl = [], -1.0, time.time(), []
    model.train()
    for step, (x, y, ig) in enumerate(dl, start=1):
        x, y, ig = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), ig.to(dev, non_blocking=True)
        set_lr(step - 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(x)
            if teacher is not None:
                with torch.no_grad():
                    soft = torch.sigmoid(teacher(x).float())
                y = 0.5 * y + 0.5 * soft
            loss = masked_bce(logits, y, ig)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl.append(loss.item())
        if step % 100 == 0:
            el = time.time() - t0
            print(
                f"step {step} loss {np.mean(tl[-100:]):.4f} {step * a.bs / el:.0f} win/s elapsed {el / 60:.1f} min",
                flush=True,
            )
        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate_sets(model, a.val, dev)
            score = float(np.mean([r["frame"]["macro_ap"] for r in res.values()]))
            rec = {
                "step": step,
                "loss": float(np.mean(tl[-a.eval_every :])),
                "val_macro_ap": score,
                "sets": {
                    k: {
                        lvl: {m: v[lvl][m] for m in ("macro_auc", "macro_ap", "macro_f1")}
                        for lvl in ("frame", "segment_1s", "clip")
                    }
                    for k, v in res.items()
                },
                "per_class_ap": {k: {c: v["frame"]["per_class"][c]["ap"] for c in CLASSES_V3} for k, v in res.items()},
            }
            hist.append(rec)
            mark = ""
            if score > best:
                best = score
                save(a.out, model, a.arch, {"step": step, "val": rec, "args": vars(a)})
                mark = " *"
            print(
                f"EVAL step {step} val frame macro-AP {score:.4f}"
                + "".join(
                    f" | {k}: AP {v['frame']['macro_ap']:.3f} segF1 {v['segment_1s']['macro_f1']:.3f}"
                    for k, v in res.items()
                )
                + mark,
                flush=True,
            )
            if a.log:
                Path(a.log).write_text(json.dumps({"history": hist, "best": best}, indent=1))
        if step >= a.steps:
            break
    print(f"done: best val macro-AP {best:.4f} -> {a.out} [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
