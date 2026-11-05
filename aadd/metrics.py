"""Metrics for per-chunk multi-label / single-hot degradation labels."""

from __future__ import annotations

import numpy as np

from aadd.constants import CLASSES, NONE_CLASS


def compute_metrics(
    y_true: np.ndarray, probs: np.ndarray, classes: list[str] = CLASSES, threshold: float = 0.5
) -> dict:
    """``y_true`` [N, C] in {0,1}, ``probs`` [N, C] in [0,1].

    Returns per-class precision / recall / F1 / support / ROC-AUC / AP,
    micro / macro / weighted / samples F1, exact-match accuracy, and, because
    every chunk carries exactly one label in the v1/v2 protocol, an argmax
    accuracy and an argmax confusion matrix.
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = (probs >= threshold).astype(int)
    C = len(classes)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_class = {}
    for i, c in enumerate(classes):
        auc = ap = float("nan")
        if 0 < y_true[:, i].sum() < len(y_true):
            auc = float(roc_auc_score(y_true[:, i], probs[:, i]))
            ap = float(average_precision_score(y_true[:, i], probs[:, i]))
        per_class[c] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
            "roc_auc": auc,
            "ap": ap,
        }
    out = {
        "threshold": threshold,
        "n": len(y_true),
        "per_class": per_class,
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float((y_pred == y_true).all(axis=1).mean()),
        "label_accuracy": float((y_pred == y_true).mean()),  # bitwise accuracy over all outputs
    }
    degr = [i for i, c in enumerate(classes) if c != NONE_CLASS]
    out["f1_macro_degradations"] = float(np.mean([f[i] for i in degr]))
    single = (y_true.sum(axis=1) == 1).all()
    out["single_label"] = bool(single)
    if single:
        t = y_true.argmax(axis=1)
        pr = probs.argmax(axis=1)
        out["argmax_accuracy"] = float((t == pr).mean())
        out["confusion_argmax"] = confusion_matrix(t, pr, labels=list(range(C))).tolist()
    return out


def metrics_table(m: dict, classes: list[str] = CLASSES) -> str:
    """Markdown table of per-class metrics plus the averages."""
    lines = ["| class | precision | recall | F1 | ROC-AUC | AP | support |", "|---|---|---|---|---|---|---|"]
    for c in classes:
        pc = m["per_class"][c]
        lines.append(
            f"| {c} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['roc_auc']:.3f} | {pc['ap']:.3f} | {pc['support']} |"
        )
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in [
        "f1_micro",
        "f1_macro",
        "f1_macro_degradations",
        "f1_weighted",
        "f1_samples",
        "exact_match",
        "label_accuracy",
        "argmax_accuracy",
    ]:
        if k in m:
            lines.append(f"| {k} | {m[k]:.4f} |")
    lines.append(f"| n_chunks | {m['n']} |")
    return "\n".join(lines)


def file_level_correlation(true_counts: np.ndarray, pred_counts: np.ndarray) -> dict:
    """Spearman / Pearson between the true number of degraded chunks per file and the predicted AADD count."""
    from scipy.stats import pearsonr, spearmanr

    t = np.asarray(true_counts, dtype=float)
    p = np.asarray(pred_counts, dtype=float)
    if len(t) < 3 or t.std() == 0 or p.std() == 0:
        return {"n_files": len(t), "spearman": float("nan"), "pearson": float("nan")}
    return {
        "n_files": len(t),
        "spearman": float(spearmanr(t, p).correlation),
        "pearson": float(pearsonr(t, p)[0]),
        "mae": float(np.abs(t - p).mean()),
    }
