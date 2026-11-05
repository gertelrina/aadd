"""Evaluate a v3 checkpoint: fixed synthetic test sets + out-of-domain sets.

python -m aadd.v3.evaluate CKPT --sets DIR --hf DIR --out REPORT.json
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from aadd.v3.data import FixedSet
from aadd.v3.infer import FrameModel
from aadd.v3.metrics import evaluate_arrays


def _decode(b) -> np.ndarray:
    import librosa
    import soundfile as sf

    if isinstance(b, dict):
        b = b.get("bytes")
    y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y


def clip_scores(fm: FrameModel, clips: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per clip: (mean frame prob, 95th-percentile frame prob) per class."""
    mean, p95 = [], []
    for y in clips:
        p = fm.frame_probs(y)
        mean.append(p.mean(0))
        p95.append(np.percentile(p, 95, axis=0))
    return np.array(mean), np.array(p95)


def ood_tasks(hf: Path):
    """Yield (task name, clips, {class: labels}) for every OOD binary task."""
    import pyarrow.parquet as pq

    t = pq.read_table(next((hf / "JacobLinCool__VoiceBank-DEMAND-16k/data").glob("test*.parquet"))).to_pylist()
    clean = [_decode(r["clean"]) for r in t]
    noisy = [_decode(r["noisy"]) for r in t]
    yield (
        "VoiceBank-DEMAND test (824 noisy vs 824 clean twins)",
        clean + noisy,
        {"noise": np.r_[np.zeros(len(clean)), np.ones(len(noisy))]},
    )

    d = hf / "nkdem__DNS-Challenge-2020-DevTest-16k/data"
    nr = pq.read_table(next(d.glob("synthetic_no_reverb*"))).to_pylist()
    wr = pq.read_table(next(d.glob("synthetic_with_reverb*"))).to_pylist()
    c_nr = [_decode(r["clean"]) for r in nr]
    n_nr = [_decode(r["noisy"]) for r in nr]
    c_wr = [_decode(r["clean"]) for r in wr]
    n_wr = [_decode(r["noisy"]) for r in wr]
    clips = c_nr + n_nr + c_wr + n_wr
    k = len(nr)
    yield (
        "DNS-2020 synthetic (clean / noisy, dry / reverberant)",
        clips,
        {
            "noise": np.r_[np.zeros(k), np.ones(k), np.zeros(k), np.ones(k)],
            "reverb": np.r_[np.zeros(2 * k), np.ones(2 * k)],
        },
    )
    for name, sub in [("noise", "NoiseDetection_VCTK_MUSAN-Noise")] + [
        ("reverb", f"ReverberationDetection_VCTK_RirsNoises-{r}") for r in ("SmallRoom", "MediumRoom", "LargeRoom")
    ]:
        t = pq.read_table(next((hf / f"DynamicSuperb__{sub}/data").glob("*.parquet"))).to_pylist()
        yield (
            f"Dynamic-SUPERB {sub}",
            [_decode(r["audio"]) for r in t],
            {name: np.array([0.0 if r["label"] == "clean" else 1.0 for r in t])},
        )


def main(argv=None):
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-synthetic", action="store_true")
    a = ap.parse_args(argv)
    fm = FrameModel(a.ckpt, a.device)
    rep = {"ckpt": a.ckpt, "synthetic": {}, "ood": {}}
    if not a.skip_synthetic:
        for p in sorted(Path(a.sets).glob("test_*.npz")):
            ds = FixedSet(str(p))
            P = np.stack(
                [fm.frame_probs(ds.X[i].astype(np.float32) / 32767.0)[: ds.Y.shape[1]] for i in range(len(ds))]
            )
            r = evaluate_arrays(ds.Y.astype(np.float32), P, ds.IG, fm.classes)
            rep["synthetic"][p.stem] = r
            print(p.stem, {lvl: round(r[lvl]["macro_ap"], 4) for lvl in ("frame", "segment_1s", "clip")}, flush=True)
    for name, clips, labels in ood_tasks(Path(a.hf)):
        mean, p95 = clip_scores(fm, clips)
        res = {"n": len(clips)}
        for c, y in labels.items():
            j = fm.classes.index(c)
            res[c] = {
                "auc_mean": float(roc_auc_score(y, mean[:, j])),
                "auc_p95": float(roc_auc_score(y, p95[:, j])),
                "recall_at_0.5_p95": float((p95[y == 1, j] >= 0.5).mean()),
                "false_alarm_at_0.5_p95": float((p95[y == 0, j] >= 0.5).mean()),
            }
        neg_all = np.all(np.stack(list(labels.values())) == 0, axis=0)
        if neg_all.any():
            fa = p95[neg_all] >= 0.5
            res["clean_clips_any_class_false_alarm"] = float(fa.any(1).mean())
            res["clean_clips_false_alarm_by_class"] = {c: float(fa[:, j].mean()) for j, c in enumerate(fm.classes)}
        rep["ood"][name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "clean_clips_false_alarm_by_class"}), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
