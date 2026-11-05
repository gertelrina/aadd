"""Inference for v3 frame models on audio of any length.

Audio is cut into 8-s windows with ``overlap`` seconds of overlap; frame
probabilities (20 ms) are averaged where windows overlap. A tail shorter than a
window is zero-padded and its padded frames are dropped.
"""

from __future__ import annotations

import numpy as np
import torch

SR = 16000
FRAME = 320


class FrameModel:
    def __init__(self, path: str, device: str | torch.device = "cpu"):
        from aadd.v3.train import load

        self.device = torch.device(device)
        self.model, self.ckpt = load(path, self.device)
        self.classes: list[str] = list(self.ckpt["classes"])
        self.win = int(self.ckpt.get("window_seconds", 8.0) * SR)

    @torch.no_grad()
    def frame_probs(self, y: np.ndarray, overlap: float = 1.0, batch: int = 16) -> np.ndarray:
        """``y`` float32 mono 16 kHz -> probabilities ``[n_frames, n_classes]`` (n_frames = ceil(len/320))."""
        n = len(y)
        nf = int(np.ceil(n / FRAME))
        hop = self.win - int(overlap * SR)
        starts = list(range(0, max(n - self.win, 0) + 1, hop))
        if starts[-1] + self.win < n:
            starts.append(n - self.win)
        wins = []
        for s in starts:
            w = y[s : s + self.win]
            wins.append(np.pad(w, (0, self.win - len(w))))
        acc = np.zeros((nf + self.win // FRAME, len(self.classes)), dtype=np.float64)
        cnt = np.zeros(len(acc), dtype=np.float64)
        use_amp = self.device.type == "cuda"
        for i in range(0, len(wins), batch):
            x = torch.from_numpy(np.stack(wins[i : i + batch])).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                p = torch.sigmoid(self.model(x).float()).cpu().numpy()
            for j, s in enumerate(starts[i : i + batch]):
                f0 = s // FRAME
                k = p.shape[1]
                # drop the 0.5 s at window edges that are not file edges (less context there)
                lo = 0 if s == 0 else min(25, k)
                hi = k
                acc[f0 + lo : f0 + hi] += p[j, lo:hi]
                cnt[f0 + lo : f0 + hi] += 1
        cnt = np.maximum(cnt, 1)
        return (acc / cnt[:, None])[:nf].astype(np.float32)


def pool_frames(probs: np.ndarray, frames_per_segment: int, how: str = "max") -> np.ndarray:
    n = int(np.ceil(len(probs) / frames_per_segment))
    out = np.zeros((n, probs.shape[1]), dtype=np.float32)
    for i in range(n):
        seg = probs[i * frames_per_segment : (i + 1) * frames_per_segment]
        out[i] = seg.max(0) if how == "max" else seg.mean(0)
    return out
