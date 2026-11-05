"""Lightweight DSP detectors for the AADD degradation classes.

They are *not* meant to replace the CNN; they are the trivial baselines a
reader will ask about ("why a neural net for clipping?") and they are useful
sanity checks on real-world data where the synthetic classes may not apply.
Every function maps a 1-s chunk to a scalar score where *higher = more likely
degraded*. Thresholds are intentionally not baked in; ``aadd eval --baselines``
reports threshold-free ROC-AUC / AP and the best achievable F1.

Coverage:

* ``ClippingDistortion`` – fraction of samples sitting exactly on the signal's
  min / max (audiomentations clips at percentile levels, so the clipped values
  repeat exactly).
* ``LowPassFilter`` – share of spectral energy *below* 1 kHz (LPF 150–750 Hz
  leaves almost nothing above).
* ``HighPassFilter`` – share of spectral energy *above* 1 kHz.
* ``AddGaussianNoise`` – high-band (4–8 kHz) noise floor in dB, estimated as
  the 10th percentile of per-frame energy; white noise raises it by 20–40 dB.
* ``TimeStretch`` – fraction of trailing (near-)zero samples: rate > 1 shortens
  the chunk and audiomentations zero-pads it back to 1 s. Only the rate > 1
  half of the class is detectable this way.
* ``Shift`` – largest sample-to-sample jump relative to the chunk RMS (a
  rolled chunk has one discontinuity, softened by a 5 ms fade).
* ``PitchShift`` – no simple detector; score is constant.
"""

from __future__ import annotations

import numpy as np

from aadd.constants import SAMPLE_RATE, SEGMENT_SECONDS

BASELINE_CLASSES = [
    "AddGaussianNoise",
    "PitchShift",
    "TimeStretch",
    "Shift",
    "ClippingDistortion",
    "LowPassFilter",
    "HighPassFilter",
]


def clipping_score(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    hi, lo = x.max(), x.min()
    if hi - lo < 1e-6:
        return 0.0
    tol = 1e-6
    n_ext = np.sum(x >= hi - tol) + np.sum(x <= lo + tol)
    return float(n_ext / len(x))


def _band_energy(x: np.ndarray, sr: int, lo: float, hi: float, n_fft: int = 1024) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=max(n_fft, len(x)))) ** 2
    freqs = np.fft.rfftfreq(max(n_fft, len(x)), 1.0 / sr)
    return float(spec[(freqs >= lo) & (freqs < hi)].sum())


def lowpass_score(x: np.ndarray, sr: int) -> float:
    tot = _band_energy(x, sr, 0, sr / 2) + 1e-12
    return _band_energy(x, sr, 0, 1000) / tot


def highpass_score(x: np.ndarray, sr: int) -> float:
    tot = _band_energy(x, sr, 0, sr / 2) + 1e-12
    return _band_energy(x, sr, 1000, sr / 2) / tot


def noise_floor_score(x: np.ndarray, sr: int, frame: int = 512, lo: float = 4000.0) -> float:
    """10th percentile over frames of the 4–8 kHz band energy, in dB."""
    if len(x) < frame:
        return -120.0
    n = len(x) // frame
    frames = x[: n * frame].reshape(n, frame) * np.hanning(frame)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    band = spec[:, freqs >= lo].mean(axis=1)
    return float(10 * np.log10(np.percentile(band, 10) + 1e-12))


def trailing_zero_score(x: np.ndarray, eps: float = 1e-5) -> float:
    nz = np.flatnonzero(np.abs(x) > eps)
    if len(nz) == 0:
        return 1.0
    return float((len(x) - 1 - nz[-1]) / len(x))


def discontinuity_score(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
    return float(np.abs(np.diff(x)).max() / rms) if len(x) > 1 else 0.0


def baseline_scores(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Scores for one chunk, in ``BASELINE_CLASSES`` order."""
    x = np.asarray(x, dtype=np.float64)
    return np.array(
        [
            noise_floor_score(x, sr),
            0.0,  # PitchShift: no baseline
            trailing_zero_score(x),
            discontinuity_score(x),
            clipping_score(x),
            lowpass_score(x, sr),
            highpass_score(x, sr),
        ],
        dtype=np.float64,
    )


def baseline_scores_file(path: str, sr: int = SAMPLE_RATE, seconds: float = SEGMENT_SECONDS) -> np.ndarray:
    """[n_chunks, len(BASELINE_CLASSES)] scores for a file (trailing partial second dropped)."""
    from aadd.features import load_audio, segment_audio

    y, sr = load_audio(path, sr=sr)
    segs = segment_audio(y, sr, seconds)
    if not segs:
        return np.zeros((0, len(BASELINE_CLASSES)))
    return np.stack([baseline_scores(s.samples, sr) for s in segs])
