"""Cross-evaluation between the model generations (writes runs/cross_eval.json).

* v3.1 on the v1/v2 synthetic test set: six v3 classes mapped to the audiomentations ones;
  a second's score is the maximum over its 50 frames; metrics at threshold 0.5.
* v1 and v2 on the public noise benchmarks: a file's noise score is the 95th percentile of its
  per-second AddGaussianNoise probability; "any degradation" is the 95th percentile of the
  per-second maximum over the seven degradation outputs.

    python scripts/cross_eval.py --test data/aadd_test_clean --hf ood_hf --out runs/cross_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MAP = {
    "AddGaussianNoise": "noise",
    "ClippingDistortion": "clipping",
    "LowPassFilter": "lowpass",
    "HighPassFilter": "highpass",
    "PitchShift": "pitch_shift",
    "TimeStretch": "time_stretch",
}


def v3_on_v2_test(test_dir: Path, device: str) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    from aadd.data.augment import read_labels
    from aadd.features import load_audio
    from aadd.infer import Detector

    det = Detector("v3", device=device, resolution=1.0)
    rows = list(csv.DictReader((test_dir / "manifest.csv").open()))
    Y, P = [], []
    for i, r in enumerate(rows):
        y, sr = load_audio(r["audio"])
        res = det.predict(y, sr)
        labels = [row["augmentation"] for row in read_labels(r["labels"])]  # one label per second
        n = min(len(labels), res.probs.shape[0])
        Y.extend(labels[:n])
        P.append(res.probs[:n])
        if i % 2000 == 0:
            print(f"v3.1 on v2 test: {i}/{len(rows)}", flush=True)
    P = np.concatenate(P)
    Y = np.array(Y)
    out = {}
    for a_cls, v3_cls in MAP.items():
        t = (Y == a_cls).astype(int)
        s = P[:, det.classes.index(v3_cls)]
        out[a_cls] = {
            "v3_class": v3_cls,
            "auc": float(roc_auc_score(t, s)),
            "f1": float(f1_score(t, s >= 0.5)),
            "precision": float(precision_score(t, s >= 0.5)),
            "recall": float(recall_score(t, s >= 0.5)),
            "support": int(t.sum()),
        }
    clean = Y == "None"
    out["_clean_seconds_any_class"] = float((P[clean] >= 0.5).any(1).mean())
    out["_n_chunks"] = len(Y)
    return out


def old_models_on_benchmarks(hf: Path, device: str) -> dict:
    from sklearn.metrics import roc_auc_score

    from aadd.constants import CLASSES, NONE_CLASS
    from aadd.infer import Detector
    from aadd.v3.evaluate import ood_tasks

    deg = [i for i, c in enumerate(CLASSES) if c != NONE_CLASS]
    j_noise = CLASSES.index("AddGaussianNoise")
    out = {}
    for model in ("v1", "v2"):
        det = Detector(model, device=device)
        out[model] = {}
        for name, clips, labels in ood_tasks(hf):
            if "noise" not in labels:
                continue
            y = labels["noise"]
            noise, anyd = [], []
            for x in clips:
                p = det.predict(x, 16000).probs
                noise.append(np.percentile(p[:, j_noise], 95))
                anyd.append(np.percentile(p[:, deg].max(1), 95))
            noise, anyd = np.array(noise), np.array(anyd)
            out[model][name] = {
                "noise_auc": float(roc_auc_score(y, noise)),
                "any_degradation_auc": float(roc_auc_score(y, anyd)),
                "noise_recall_at_0.5": float((noise[y == 1] >= 0.5).mean()),
                "false_alarm_clean_at_0.5": float((anyd[y == 0] >= 0.5).mean()),
            }
            print(model, name, out[model][name], flush=True)
    return {"v1_on_real_noise": out["v1"], "v2_on_real_noise": out["v2"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="v1/v2 test set folder (aadd make-data)")
    ap.add_argument("--hf", required=True, help="folder with the Hugging Face benchmark snapshots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    rep = {"v3.1_on_v2_test": v3_on_v2_test(Path(a.test), a.device)}
    rep.update(old_models_on_benchmarks(Path(a.hf), a.device))
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
