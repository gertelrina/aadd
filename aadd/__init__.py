"""AADD — Automatic Audio Degradation Detection.

Finds which degradation is present in speech, and where: v3.1 predicts 12 classes
(noise, music, reverberation, clipping, low-pass, high-pass, codec artefacts, packet
loss, hum, clicks, pitch shift, time stretch) every 20 ms; the earlier v1 / v2 models
predict 7 synthetic classes per second.
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

__version__ = "3.1.0"
