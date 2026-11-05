"""AADD — Automatic Audio Degradation Detection.

Per-second detection of synthetic audio degradations (Gaussian noise, pitch shift,
time stretch, circular shift, clipping, low-pass, high-pass) in speech recordings.
"""

from aadd.constants import CLASSES, DEGRADATION_CLASSES, SAMPLE_RATE, SEGMENT_SECONDS
from aadd.infer import Detector, FileResult

__all__ = [
    "CLASSES",
    "DEGRADATION_CLASSES",
    "SAMPLE_RATE",
    "SEGMENT_SECONDS",
    "Detector",
    "FileResult",
    "__version__",
]

__version__ = "2.0.0"
