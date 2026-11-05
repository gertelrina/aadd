"""Inference API: ``Detector`` turns audio into per-second degradation probabilities.

>>> from aadd import Detector
>>> det = Detector("v2")              # or "v1", or a path to a checkpoint
>>> res = det.predict_file("x.flac")
>>> res.detected                      # {'ClippingDistortion': 0.93, ...}  (file level)
>>> res.probs.shape                   # (n_seconds, 8)
>>> res.degradation_count             # "AADD count": # of (second, class) detections
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aadd.constants import NONE_CLASS
from aadd.features import FeatureSpec, load_audio, segment_audio
from aadd.model import DEFAULT_MODEL, checkpoint_format, load_checkpoint, pick_device, resolve_model_path

AUDIO_EXTENSIONS = (".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif")


@dataclass
class FileResult:
    path: str
    sample_rate: int
    duration: float
    starts: np.ndarray  # [n] segment start times (s)
    ends: np.ndarray  # [n] segment end times (s)
    probs: np.ndarray  # [n, len(classes)] sigmoid outputs
    classes: list[str]
    threshold: float
    model: str
    error: str | None = None

    # ----- derived quantities ---------------------------------------------
    @property
    def n_segments(self) -> int:
        return int(self.probs.shape[0])

    @property
    def degradation_indices(self) -> list[int]:
        return [i for i, c in enumerate(self.classes) if c != NONE_CLASS]

    @property
    def binary(self) -> np.ndarray:
        """[n, C] boolean detections at ``threshold`` (the None column is kept but ignored elsewhere)."""
        return self.probs >= self.threshold

    @property
    def segment_degraded(self) -> np.ndarray:
        """[n] True where at least one degradation class fires."""
        if self.n_segments == 0:
            return np.zeros(0, dtype=bool)
        return self.binary[:, self.degradation_indices].any(axis=1)

    @property
    def degraded_segments(self) -> int:
        return int(self.segment_degraded.sum())

    @property
    def degraded_fraction(self) -> float:
        return float(self.segment_degraded.mean()) if self.n_segments else 0.0

    @property
    def degradation_count(self) -> int:
        """Total number of (segment, class) detections (the 'AADD count')."""
        if self.n_segments == 0:
            return 0
        return int(self.binary[:, self.degradation_indices].sum())

    @property
    def max_probs(self) -> dict[str, float]:
        if self.n_segments == 0:
            return {c: 0.0 for c in self.classes}
        m = self.probs.max(axis=0)
        return {c: float(m[i]) for i, c in enumerate(self.classes)}

    @property
    def detected(self) -> dict[str, float]:
        """File-level verdict: degradation classes whose max probability over segments >= threshold."""
        return {c: p for c, p in self.max_probs.items() if c != NONE_CLASS and p >= self.threshold}

    @property
    def is_clean(self) -> bool:
        return self.error is None and not self.detected

    def segment_labels(self) -> list[list[str]]:
        b = self.binary
        return [[self.classes[j] for j in self.degradation_indices if b[i, j]] for i in range(self.n_segments)]

    # ----- tabular export ---------------------------------------------------
    def file_row(self) -> dict:
        row = {
            "path": self.path,
            "duration_s": round(self.duration, 3),
            "n_segments": self.n_segments,
            "degraded_segments": self.degraded_segments,
            "degraded_fraction": round(self.degraded_fraction, 4),
            "aadd_count": self.degradation_count,
            "detected": ";".join(sorted(self.detected)),
            "status": "error" if self.error else ("clean" if self.is_clean else "degraded"),
            "error": self.error or "",
        }
        for c, p in self.max_probs.items():
            row[f"p_{c}"] = round(p, 4)
        return row

    def segment_rows(self) -> list[dict]:
        rows = []
        labels = self.segment_labels()
        for i in range(self.n_segments):
            r = {
                "path": self.path,
                "start_s": float(self.starts[i]),
                "end_s": float(self.ends[i]),
                "labels": ";".join(labels[i]),
            }
            for j, c in enumerate(self.classes):
                r[f"p_{c}"] = round(float(self.probs[i, j]), 4)
            rows.append(r)
        return rows

    def to_dict(self, frame_hop: float = 0.1) -> dict:
        d = {
            "path": self.path,
            "model": self.model,
            "threshold": self.threshold,
            "sample_rate": self.sample_rate,
            "duration_s": self.duration,
            "classes": self.classes,
            "file": self.file_row(),
            "segments": self.segment_rows(),
        }
        fp = getattr(self, "frame_probs", None)
        if fp is not None and len(fp):
            # frame models: keep a finer timeline (max over ``frame_hop`` s), one byte per value
            import base64

            k = max(1, round(frame_hop / 0.02))
            n = int(np.ceil(len(fp) / k))
            pooled = np.stack([fp[i * k : (i + 1) * k].max(0) for i in range(n)])
            q = np.clip(np.round(pooled * 255), 0, 255).astype(np.uint8)
            d["frames"] = {"hop_s": k * 0.02, "n": n, "q": base64.b64encode(q.tobytes()).decode()}
        return d

    @classmethod
    def failed(cls, path: str, error: str, classes: list[str], threshold: float, model: str) -> FileResult:
        z = np.zeros((0, len(classes)), dtype=np.float32)
        return cls(path, 0, 0.0, np.zeros(0), np.zeros(0), z, list(classes), threshold, model, error=error)


class Detector:
    """Load a model once and score files / arrays.

    Parameters
    ----------
    model : registry name (``"v1"``, ``"v2"``) or a checkpoint path.
    device : ``"cpu"``, ``"cuda"``, ``"mps"`` or ``None`` (auto).
    threshold : probability threshold for a detection.
    keep_tail : keep (zero-pad) a trailing partial second >= 0.25 s. The v1
        pipeline dropped it; ``False`` by default for comparability.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        threshold: float = 0.5,
        batch_size: int = 64,
        keep_tail: bool = False,
        resolution: float = 1.0,
    ):
        self.model_name = model
        self.device = pick_device(device)
        self.resolution = float(resolution)
        self.frame_model = None
        if checkpoint_format(model).startswith("aadd-v3"):
            from aadd.v3.infer import FrameModel

            self.frame_model = FrameModel(str(resolve_model_path(model)), self.device)
            self.model, self.ckpt = self.frame_model.model, self.frame_model.ckpt
            self.classes = list(self.frame_model.classes)
            self.features = FeatureSpec(kind="frames")
        else:
            self.model, self.ckpt = load_checkpoint(model, self.device)
            self.classes = list(self.ckpt["classes"])
            self.features = FeatureSpec.from_dict(self.ckpt["features"])
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.keep_tail = keep_tail

    # ----- core ---------------------------------------------------------------
    @torch.no_grad()
    def predict_batch(self, feats: np.ndarray) -> np.ndarray:
        """``feats`` [n, C, H, W] float32 -> sigmoid probabilities [n, classes]."""
        out = []
        for i in range(0, len(feats), self.batch_size):
            x = torch.from_numpy(np.ascontiguousarray(feats[i : i + self.batch_size])).to(self.device)
            out.append(torch.sigmoid(self.model(x)).float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(self.classes)), dtype=np.float32)

    def featurize(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Audio -> (starts, ends, feats). Resamples to the model's sample rate if needed."""
        if sr != self.features.sample_rate:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=self.features.sample_rate)
            sr = self.features.sample_rate
        segs = segment_audio(y, sr, self.features.segment_seconds, keep_tail=self.keep_tail)
        feats = np.stack([self.features(s.samples, sr) for s in segs]) if segs else np.zeros((0,), dtype=np.float32)
        starts = np.array([s.start for s in segs], dtype=np.float64)
        ends = np.array([s.end for s in segs], dtype=np.float64)
        return starts, ends, feats

    def predict(self, y: np.ndarray, sr: int, path: str = "<array>") -> FileResult:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:  # (channels, n) or (n, channels)
            y = y.mean(axis=0 if y.shape[0] <= 8 else 1)
        if self.frame_model is not None:
            return self._predict_frames(y, sr, path)
        starts, ends, feats = self.featurize(y, sr)
        probs = self.predict_batch(feats) if len(starts) else np.zeros((0, len(self.classes)), dtype=np.float32)
        return FileResult(path, sr, len(y) / sr, starts, ends, probs, self.classes, self.threshold, self.model_name)

    def _predict_frames(self, y: np.ndarray, sr: int, path: str) -> FileResult:
        from aadd.v3.infer import pool_frames

        if sr != 16000:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        dur = len(y) / 16000
        fp = self.frame_model.frame_probs(y) if len(y) else np.zeros((0, len(self.classes)), dtype=np.float32)
        k = max(1, round(self.resolution / 0.02))
        probs = pool_frames(fp, k) if len(fp) else fp
        starts = np.arange(len(probs)) * self.resolution
        ends = np.minimum(starts + self.resolution, dur)
        res = FileResult(path, sr, dur, starts, ends, probs, self.classes, self.threshold, self.model_name)
        res.frame_probs = fp
        return res

    def predict_file(self, path: str | os.PathLike) -> FileResult:
        path = str(path)
        try:
            y, sr = load_audio(path, sr=self.features.sample_rate)
            return self.predict(y, sr, path=path)
        except Exception as e:  # noqa: BLE001 - we want a per-file error row, not a crash
            return FileResult.failed(path, f"{type(e).__name__}: {e}", self.classes, self.threshold, self.model_name)

    def scan(
        self, paths: Iterable[str | os.PathLike], workers: int = 0, progress: bool = False
    ) -> Iterator[FileResult]:
        """Score many files. ``workers > 0`` featurises in parallel processes (useful for v1 'png' features)."""
        paths = [str(p) for p in paths]
        if self.frame_model is not None:
            workers = 0  # frame models featurise on the device
        it: Iterable = paths
        if progress:
            from tqdm import tqdm

            it = tqdm(paths, unit="file")
        if workers <= 0:
            for p in it:
                yield self.predict_file(p)
            return
        spec = self.features
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = ex.map(
                _featurize_worker, paths, [spec.to_dict()] * len(paths), [self.keep_tail] * len(paths), chunksize=4
            )
            for p, (starts, ends, feats, sr, dur, err) in zip(it, futs):
                if err:
                    yield FileResult.failed(p, err, self.classes, self.threshold, self.model_name)
                    continue
                probs = self.predict_batch(feats) if len(starts) else np.zeros((0, len(self.classes)), dtype=np.float32)
                yield FileResult(p, sr, dur, starts, ends, probs, self.classes, self.threshold, self.model_name)


def _featurize_worker(path: str, spec_dict: dict, keep_tail: bool):
    try:
        spec = FeatureSpec.from_dict(spec_dict)
        y, sr = load_audio(path, sr=spec.sample_rate)
        segs = segment_audio(y, sr, spec.segment_seconds, keep_tail=keep_tail)
        feats = np.stack([spec(s.samples, sr) for s in segs]) if segs else np.zeros((0,), dtype=np.float32)
        starts = np.array([s.start for s in segs], dtype=np.float64)
        ends = np.array([s.end for s in segs], dtype=np.float64)
        return starts, ends, feats, sr, len(y) / sr, None
    except Exception as e:  # noqa: BLE001
        return None, None, None, 0, 0.0, f"{type(e).__name__}: {e}"


def find_audio_files(
    paths: Sequence[str | os.PathLike], recursive: bool = True, extensions: Sequence[str] = AUDIO_EXTENSIONS
) -> list[str]:
    """Expand files / directories into a sorted list of audio file paths."""
    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in extensions)
    out: list[str] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out.extend(str(f) for f in it if f.is_file() and f.suffix.lower() in exts)
        elif p.is_file():
            out.append(str(p))
        else:
            raise FileNotFoundError(str(p))
    return sorted(dict.fromkeys(out))
