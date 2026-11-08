"""Optional NISQA quality scores next to AADD's detections (``aadd scan --nisqa``).

NISQA (Mittag et al., Interspeech 2021) predicts an overall MOS and four quality dimensions,
each from 1 (bad) to 5 (excellent). It says how good a recording sounds, AADD says what is
wrong and where; together they show both. The implementation and weights come from
torchmetrics (``pip install "aadd[nisqa]"``); the NISQA weights are CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import numpy as np

KEYS = ("mos", "noisiness", "discontinuity", "colouration", "loudness")


def nisqa_scores(y: np.ndarray, sr: int) -> dict[str, float]:
    """NISQA scores of one mono signal: {mos, noisiness, discontinuity, colouration, loudness}."""
    try:
        import torch
        from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise ImportError('NISQA needs torchmetrics: pip install "aadd[nisqa]"') from e
    if len(y) < sr // 2:  # NISQA needs at least ~0.5 s
        return {k: float("nan") for k in KEYS}
    x = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    s = non_intrusive_speech_quality_assessment(x, sr).tolist()
    return {k: round(float(v), 3) for k, v in zip(KEYS, s)}
