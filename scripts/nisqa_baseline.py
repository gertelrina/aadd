"""NISQA as a reference on the out-of-domain sets AADD is evaluated on.

NISQA (Mittag et al., Interspeech 2021) predicts an overall MOS and four quality dimensions
(noisiness, discontinuity, colouration, loudness), 1 = bad, 5 = good. It says how bad a
recording is, not what is wrong, so here it is scored as a detector: ROC-AUC of (5 - score)
for "degraded vs clean", with the overall MOS and with the dimension closest to the class.

    pip install "aadd[nisqa]"                # NISQA weights: CC BY-NC-SA 4.0, fetched by torchmetrics
    python scripts/nisqa_baseline.py --hf DIR --out runs/nisqa_ood.json
    python scripts/nisqa_baseline.py --synthetic data/test --out runs/nisqa_test_clean.json

DIR holds the Hugging Face dataset snapshots used by `aadd.v3.evaluate`; `--synthetic` takes the
v1/v2 test set built by `aadd make-data` and scores NISQA the way the file-level metrics of v1/v2
are scored (true number of degraded seconds per file).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

DIMS = ("mos", "noisiness", "discontinuity", "colouration", "loudness")
MATCH = {"noise": "noisiness", "reverb": "colouration"}


def nisqa_scores(clips: list[np.ndarray], sr: int = 16000) -> np.ndarray:
    from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment as nisqa

    torch.set_num_threads(4)
    return np.array([nisqa(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), sr).tolist() for y in clips])


def synthetic(root: Path, out: Path) -> None:
    import csv

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    from aadd.features import load_audio

    rows = list(csv.DictReader((root / "manifest.csv").open()))
    t = time.time()
    s = []
    for i, r in enumerate(rows):
        p = Path(r["audio"])
        s.append(nisqa_scores([load_audio(str(p if p.exists() else root.parent.parent / p))[0]])[0])
        if i % 2000 == 0:
            print(f"{i}/{len(rows)} {time.time() - t:.0f}s", flush=True)
    s = np.array(s)
    n_deg = np.array([int(r["n_degraded"]) for r in rows])
    rep = {
        "model": "NISQA v2.0 (torchmetrics)",
        "n_files": len(rows),
        "file_level": {
            "spearman_n_degraded_vs_5_minus_mos": float(spearmanr(n_deg, 5 - s[:, 0]).statistic),
            "pearson_n_degraded_vs_5_minus_mos": float(pearsonr(n_deg, 5 - s[:, 0]).statistic),
            "auc_any_degraded_vs_clean_copy": float(roc_auc_score(n_deg > 0, 5 - s[:, 0])),
            "mean_mos_clean_copy": float(s[n_deg == 0, 0].mean()),
            "mean_mos_degraded": float(s[n_deg > 0, 0].mean()),
        },
        "seconds": round(time.time() - t, 1),
    }
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


def main() -> None:
    from sklearn.metrics import roc_auc_score

    from aadd.v3.evaluate import ood_tasks

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf")
    ap.add_argument("--synthetic", help="v1/v2 test set folder with manifest.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.synthetic:
        synthetic(Path(a.synthetic), Path(a.out))
        return
    rep = {"model": "NISQA v2.0 (torchmetrics)", "score_dims": DIMS, "ood": {}}
    for name, clips, labels in ood_tasks(Path(a.hf)):
        t = time.time()
        s = nisqa_scores(clips)
        res = {"n": len(clips), "seconds": round(time.time() - t, 1)}
        for c, y in labels.items():
            j = DIMS.index(MATCH[c])
            res[c] = {
                "auc_mos": float(roc_auc_score(y, 5 - s[:, 0])),
                f"auc_{MATCH[c]}": float(roc_auc_score(y, 5 - s[:, j])),
                "mean_mos_degraded": float(s[y == 1, 0].mean()),
                "mean_mos_clean": float(s[y == 0, 0].mean()),
            }
        rep["ood"][name] = res
        print(name, json.dumps(res), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
