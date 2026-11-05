"""Evaluation of a model (or the DSP baselines) on a ``make-data`` directory or a featurized ``.npz``.

Two paths:

* :func:`evaluate_dir` runs the full inference pipeline (audio -> features ->
  model) file by file through :class:`aadd.infer.Detector`, so v1 (PNG) and v2
  (mel) are scored by exactly the code that users run. It also computes the
  file-level "AADD count vs. true number of degraded seconds" correlation the
  count of degraded seconds per file.
* :func:`evaluate_npz` scores precomputed features (fast; used during training).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from aadd.constants import CLASSES
from aadd.data.augment import read_labels
from aadd.data.dataset import ChunkArrayDataset, labels_to_targets, load_manifest
from aadd.metrics import compute_metrics, file_level_correlation, metrics_table


def evaluate_dir(
    model: str,
    data_dir: str | os.PathLike,
    threshold: float = 0.5,
    workers: int = 0,
    limit: int | None = None,
    device: str | None = None,
    batch_size: int = 64,
    progress: bool = True,
    save_predictions: str | None = None,
) -> dict:
    from aadd.infer import Detector

    rows = load_manifest(data_dir)
    if limit:
        rows = rows[:limit]
    det = Detector(model, device=device, threshold=threshold, batch_size=batch_size)
    classes = det.classes
    ys, ps, true_counts, pred_counts, pred_rows = [], [], [], [], []
    n_err = 0
    for r, res in zip(rows, det.scan([r["audio"] for r in rows], workers=workers, progress=progress)):
        if res.error:
            n_err += 1
            continue
        lab = read_labels(r["labels"])
        n = min(res.n_segments, len(lab))
        if n == 0:
            continue
        y = labels_to_targets(lab[:n], classes)
        ys.append(y)
        ps.append(res.probs[:n])
        true_counts.append(int(y[:, [i for i, c in enumerate(classes) if c != "None"]].sum()))
        pred_counts.append(res.degradation_count)
        if save_predictions:
            for i in range(n):
                pred_rows.append([r["audio"], i, lab[i]["augmentation"], *np.round(res.probs[i], 4).tolist()])
    y_true = np.concatenate(ys)
    probs = np.concatenate(ps)
    m = compute_metrics(y_true, probs, classes, threshold)
    out = {
        "model": model,
        "features": det.features.to_dict(),
        "data_dir": str(data_dir),
        "n_files": len(ys),
        "n_errors": n_err,
        "chunk_metrics": m,
        "file_level": file_level_correlation(np.array(true_counts), np.array(pred_counts)),
    }
    if save_predictions:
        import csv

        with open(save_predictions, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["audio", "chunk", "label", *[f"p_{c}" for c in classes]])
            w.writerows(pred_rows)
    return out


def evaluate_npz(
    model: str, npz_path: str | os.PathLike, threshold: float = 0.5, device: str | None = None, batch_size: int = 256
) -> dict:
    from aadd.model import load_checkpoint, pick_device
    from aadd.train import predict_dataset

    dev = pick_device(device)
    net, ckpt = load_checkpoint(model, dev)
    ds = ChunkArrayDataset(npz_path)
    if ckpt["features"]["kind"] != ds.spec.kind:
        raise ValueError(f"model expects {ckpt['features']['kind']} features but {npz_path} holds {ds.spec.kind}")
    y, p = predict_dataset(net, ds, dev, batch_size=batch_size)
    m = compute_metrics(y, p, ds.classes, threshold)
    # file-level counts from the stored file index
    degr = [i for i, c in enumerate(ds.classes) if c != "None"]
    tc = np.bincount(ds.file_idx, weights=y[:, degr].sum(1))
    pc = np.bincount(ds.file_idx, weights=(p[:, degr] >= threshold).sum(1))
    return {
        "model": model,
        "features": ckpt["features"],
        "data": str(npz_path),
        "n_files": len(ds.files),
        "chunk_metrics": m,
        "file_level": file_level_correlation(tc, pc),
    }


def evaluate_baselines(
    data_dir: str | os.PathLike, limit: int | None = None, workers: int | None = None, progress: bool = True
) -> dict:
    """Threshold-free (ROC-AUC / AP) and best-F1 scores of the DSP baselines per class."""
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    from aadd.baselines import BASELINE_CLASSES

    rows = load_manifest(data_dir)
    if limit:
        rows = rows[:limit]
    jobs = [(r["audio"], r["labels"]) for r in rows]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    S, Y = [], []
    it = None
    if progress:
        from tqdm import tqdm

        it = tqdm(total=len(jobs), unit="file", desc="baselines")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_baseline_job, jobs, chunksize=8):
            if it:
                it.update(1)
            if res is None:
                continue
            s, y = res
            S.append(s)
            Y.append(y)
    if it:
        it.close()
    S = np.concatenate(S)
    Y = np.concatenate(Y)
    out = {"data_dir": str(data_dir), "n_chunks": len(Y), "per_class": {}}
    for j, c in enumerate(BASELINE_CLASSES):
        ci = CLASSES.index(c)
        y = Y[:, ci]
        s = S[:, j]
        if y.sum() == 0 or np.isnan(s).all():
            continue
        s = np.nan_to_num(s, nan=np.nanmin(s))
        pr, rc, th = precision_recall_curve(y, s)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
        k = int(np.nanargmax(f1[:-1])) if len(th) else 0
        out["per_class"][c] = {
            "roc_auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
            "best_f1": float(f1[k]),
            "best_threshold": float(th[k]) if len(th) else float("nan"),
            "support": int(y.sum()),
        }
    return out


def _baseline_job(args):
    from aadd.baselines import baseline_scores_file

    audio, labels = args
    try:
        s = baseline_scores_file(audio)
        lab = read_labels(labels)
        n = min(len(s), len(lab))
        if n == 0:
            return None
        return s[:n], labels_to_targets(lab[:n])
    except Exception:  # noqa: BLE001
        return None


def write_report(result: dict, path: str | os.PathLike, title: str | None = None) -> None:
    """Write ``<path>.json`` and ``<path>.md``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=1)
    lines = [f"# {title or 'AADD evaluation'}", ""]
    if "chunk_metrics" in result:
        lines += [
            f"model: `{result['model']}`  features: `{result['features']['kind']}`  data: `{result.get('data_dir', result.get('data'))}`  files: {result['n_files']}",
            "",
            metrics_table(result["chunk_metrics"], result["chunk_metrics"]["per_class"].keys()),
            "",
            "File level (true # degraded seconds vs predicted AADD count): "
            + ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in result["file_level"].items()
            ),
        ]
    if "per_class" in result and "chunk_metrics" not in result:
        lines += ["| class | ROC-AUC | AP | best F1 | support |", "|---|---|---|---|---|"]
        for c, v in result["per_class"].items():
            lines.append(f"| {c} | {v['roc_auc']:.3f} | {v['ap']:.3f} | {v['best_f1']:.3f} | {v['support']} |")
    with open(path.with_suffix(".md"), "w") as f:
        f.write("\n".join(lines) + "\n")
