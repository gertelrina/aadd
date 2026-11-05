"""Shared constants. The class order is fixed by the v1 checkpoint and must not change."""

CLASSES = [
    "AddGaussianNoise",
    "PitchShift",
    "TimeStretch",
    "Shift",
    "ClippingDistortion",
    "LowPassFilter",
    "HighPassFilter",
    "None",
]
NONE_CLASS = "None"
NONE_INDEX = CLASSES.index(NONE_CLASS)
DEGRADATION_CLASSES = [c for c in CLASSES if c != NONE_CLASS]
NUM_CLASSES = len(CLASSES)

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 1.0
N_MELS = 128

# Augmentation parameters used to build the v1/v2 training data. Kept here so
# that make-data, docs and tests share one definition.
AUGMENT_PARAMS = {
    "AddGaussianNoise": {"min_amplitude": 0.01, "max_amplitude": 0.05},
    "PitchShift": {"min_semitones": -8, "max_semitones": 8},
    "TimeStretch": {"min_rate": 0.5, "max_rate": 1.5},
    "Shift": {"min_shift": -1.0, "max_shift": 1.0},
    "ClippingDistortion": {"min_percentile_threshold": 0, "max_percentile_threshold": 20},
    "LowPassFilter": {"min_cutoff_freq": 150, "max_cutoff_freq": 750},
    "HighPassFilter": {"min_cutoff_freq": 2000, "max_cutoff_freq": 8000},
}
AUGMENT_PROBABILITY = 0.5  # probability that a 1-s chunk gets any degradation at all
