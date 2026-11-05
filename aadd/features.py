"""Feature extraction.

Two feature families are supported:

* ``png`` (v1) — the original pipeline: a 128-band mel spectrogram is drawn with
  ``librosa.display.specshow`` on a 4x5-inch matplotlib figure, saved as a PNG
  with a tight bounding box and read back as an RGB image (310x385 px at
  16 kHz). This reproduces the training data of the shipped v1 checkpoint
  bit-for-bit, so v1 keeps working; it is slow (~60 ms per second of audio)
  and depends on matplotlib's rasteriser.
* ``mel`` (v2) — a plain log-mel tensor in [0, 1], no image rendering. Fast,
  deterministic, and what the v2 model is trained on.

All functions take float32 mono audio in [-1, 1].
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass, field

import librosa
import numpy as np
from PIL import Image

from aadd.constants import N_MELS, SAMPLE_RATE, SEGMENT_SECONDS

# --------------------------------------------------------------------------- #
# Audio loading and segmentation
# --------------------------------------------------------------------------- #


def load_audio(path: str, sr: int | None = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load a file as float32 mono. ``sr=None`` keeps the native rate.

    The models are trained at 16 kHz, so the default resamples everything to
    16 kHz; passing a different rate silently changes the time/frequency grid.
    """
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32, copy=False), int(sr_out)


@dataclass
class Segment:
    start: float
    end: float
    samples: np.ndarray
    padded: int = 0  # number of zero samples appended (only for a short tail)


def segment_audio(
    y: np.ndarray,
    sr: int,
    seconds: float = SEGMENT_SECONDS,
    keep_tail: bool = False,
) -> list[Segment]:
    """Cut audio into non-overlapping fixed-length chunks.

    The v1 pipeline drops the trailing partial second (``keep_tail=False``).
    With ``keep_tail=True`` a partial tail of at least 0.25 s is zero-padded
    and kept. A file shorter than one chunk always yields one padded chunk.
    """
    n = round(sr * seconds)
    n_full = len(y) // n
    segments = []
    for i in range(n_full):
        segments.append(Segment(i * seconds, (i + 1) * seconds, y[i * n : (i + 1) * n]))
    rest = len(y) - n_full * n
    if n_full == 0 or (keep_tail and rest >= n // 4):
        tail = y[n_full * n :]
        pad = n - len(tail)
        segments.append(
            Segment(n_full * seconds, (n_full * seconds) + len(tail) / sr, np.pad(tail, (0, pad)), padded=pad)
        )
    return segments


# --------------------------------------------------------------------------- #
# v1: matplotlib-rendered mel image (exact replica of the original v1 code)
# --------------------------------------------------------------------------- #

V1_FIGSIZE = (4, 5)
V1_DPI = 100


def mel_db_v1(y: np.ndarray, sr: int, n_mels: int = N_MELS) -> np.ndarray:
    """Mel spectrogram in dB exactly as in v1 (librosa defaults:
    n_fft=2048, hop_length=512, power=2, ref=max, top_db=80)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)


def render_mel_png(S_dB: np.ndarray, sr: int) -> Image.Image:
    """Draw ``S_dB`` the way the v1 training data was drawn and return an RGB image.

    Uses the object-oriented matplotlib API (thread-safe, no global state) and
    an in-memory buffer instead of a temp file; the pixels are identical to the
    original ``plt.figure(figsize=(4, 5)); specshow(...); plt.axis('off');
    plt.savefig(..., bbox_inches='tight', pad_inches=0)``.
    """
    import librosa.display
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=V1_FIGSIZE, dpi=V1_DPI)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, ax=ax)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=V1_DPI, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def png_features(y: np.ndarray, sr: int) -> np.ndarray:
    """1-s chunk -> float32 CHW array in [0, 1] (torchvision ``ToTensor`` semantics)."""
    img = render_mel_png(mel_db_v1(y, sr), sr)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


# --------------------------------------------------------------------------- #
# v2: direct log-mel tensor
# --------------------------------------------------------------------------- #

V2_N_FFT = 1024
V2_HOP = 256
V2_TOP_DB = 80.0


def mel_features(
    y: np.ndarray,
    sr: int,
    n_mels: int = N_MELS,
    n_fft: int = V2_N_FFT,
    hop_length: int = V2_HOP,
    top_db: float = V2_TOP_DB,
) -> np.ndarray:
    """1-s chunk -> float32 array [1, n_mels, T] in [0, 1].

    Log-mel relative to the chunk maximum, clipped at ``-top_db`` and mapped to
    [0, 1]. With the defaults (n_fft=1024, hop=256, the usual speech setting) a 16 kHz second gives T=63.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max, top_db=top_db)
    x = (S_dB + top_db) / top_db
    return np.ascontiguousarray(x[None].astype(np.float32))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

FEATURE_FUNCS = {"png": png_features, "mel": mel_features}


@dataclass
class FeatureSpec:
    kind: str = "mel"  # 'png' | 'mel'
    sample_rate: int = SAMPLE_RATE
    segment_seconds: float = SEGMENT_SECONDS
    n_mels: int = N_MELS
    params: dict = field(default_factory=dict)

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        if self.kind == "frames":
            raise RuntimeError("frame models (v3) compute features inside the model")
        fn = FEATURE_FUNCS[self.kind]
        if self.kind == "mel":
            return fn(y, sr, n_mels=self.n_mels, **self.params)
        return fn(y, sr)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "sample_rate": self.sample_rate,
            "segment_seconds": self.segment_seconds,
            "n_mels": self.n_mels,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSpec:
        return cls(
            kind=d.get("kind", "mel"),
            sample_rate=int(d.get("sample_rate", SAMPLE_RATE)),
            segment_seconds=float(d.get("segment_seconds", SEGMENT_SECONDS)),
            n_mels=int(d.get("n_mels", N_MELS)),
            params=dict(d.get("params", {})),
        )


def iter_segment_features(
    y: np.ndarray, sr: int, spec: FeatureSpec, keep_tail: bool = False
) -> Iterator[tuple[Segment, np.ndarray]]:
    for seg in segment_audio(y, sr, spec.segment_seconds, keep_tail=keep_tail):
        yield seg, spec(seg.samples, sr)
