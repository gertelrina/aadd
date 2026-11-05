"""Frame / segment / clip metrics for v3 (multi-label with an ignore mask)."""

from __future__ import annotations

import numpy as np


def _per_class(
    y: np.ndarray,
    p: np.ndarray,
    keep: np.ndarray,
    classes: list[str],
    thr: float = 0.5,
    max_n: int = 2_000_000,
    seed: int = 0,
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    out = {}
    for j, c in enumerate(classes):
        k = keep[:, j]
        yt, pp = y[k, j], p[k, j]
        if len(yt) > max_n:
            sel = rng.choice(len(yt), max_n, replace=False)
            yt, pp = yt[sel], pp[sel]
        pos = int(yt.sum())
        if pos == 0 or pos == len(yt):
            out[c] = {
                "support": pos,
                "auc": float("nan"),
                "ap": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
            }
            continue
        pr = pp >= thr
        tp = float((pr & (yt == 1)).sum())
        prec = tp / max(pr.sum(), 1)
        rec = tp / pos
        out[c] = {
            "support": pos,
            "auc": float(roc_auc_score(yt, pp)),
            "ap": float(average_precision_score(yt, pp)),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
            "precision": float(prec),
            "recall": float(rec),
        }
    return out


def _summary(pc: dict) -> dict:
    def m(k):
        v = [x[k] for x in pc.values() if x[k] == x[k]]
        return float(np.mean(v)) if v else float("nan")

    return {"macro_auc": m("auc"), "macro_ap": m("ap"), "macro_f1": m("f1")}


def evaluate_arrays(
    Y: np.ndarray, P: np.ndarray, IG: np.ndarray, classes: list[str], thr: float = 0.5, seg_frames: int = 50
) -> dict:
    """``Y`` [N, T, C] {0,1}, ``P`` [N, T, C] probs, ``IG`` [N, T, C] bool.

    * frame level (20 ms): ignored frames excluded;
    * segment level (``seg_frames`` frames = 1 s): positive if any non-ignored frame is
      positive, score = max prob; a segment whose positives are all ignored is excluded;
    * clip level (whole window): positive if any frame positive, score = max prob.
    """
    N, T, C = Y.shape
    keep = ~IG
    fr = _per_class(Y.reshape(-1, C), P.reshape(-1, C), keep.reshape(-1, C), classes, thr)
    S = T // seg_frames
    Ys = (Y[:, : S * seg_frames] * keep[:, : S * seg_frames]).reshape(N, S, seg_frames, C).max(2)
    anyk = keep[:, : S * seg_frames].reshape(N, S, seg_frames, C).any(2)
    Ps = P[:, : S * seg_frames].reshape(N, S, seg_frames, C).max(2)
    sg = _per_class(Ys.reshape(-1, C), Ps.reshape(-1, C), anyk.reshape(-1, C), classes, thr)
    Yc = (Y * keep).max(1)
    Pc = P.max(1)
    cl = _per_class(Yc, Pc, np.ones_like(Yc, dtype=bool), classes, thr)
    clean = Yc.sum(1) == 0
    fp_clean = float((Pc[clean] >= thr).any(1).mean()) if clean.any() else float("nan")
    return {
        "frame": {"per_class": fr, **_summary(fr)},
        "segment_1s": {"per_class": sg, **_summary(sg)},
        "clip": {
            "per_class": cl,
            **_summary(cl),
            "false_alarm_rate_on_clean_clips": fp_clean,
            "n_clean_clips": int(clean.sum()),
        },
        "n_windows": int(N),
    }


def table(res: dict, level: str = "frame") -> str:
    pc = res[level]["per_class"]
    lines = ["| class | support | AUC | AP | F1 | P | R |", "|---|---|---|---|---|---|---|"]
    for c, v in pc.items():
        lines.append(
            f"| {c} | {v['support']} | {v['auc']:.3f} | {v['ap']:.3f} | {v['f1']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} |"
        )
    s = res[level]
    lines.append(f"| **macro** | | {s['macro_auc']:.3f} | {s['macro_ap']:.3f} | {s['macro_f1']:.3f} | | |")
    return "\n".join(lines)
