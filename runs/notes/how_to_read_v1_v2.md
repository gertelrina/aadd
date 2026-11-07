* `v2` is trained on the same protocol and data size as `v1`; the gains come from
  the direct log-mel input (finer time resolution: hop 256 vs 512), SpecAugment
  masking and proper model selection, not from more data.
* `Shift` (a circular roll of the samples inside the second) is invisible to `v1`
  (F1 0.000, which pulls its macro-F1 far below micro-F1). `v2` detects it (F1 0.73, AUC 0.98) through the seam the roll leaves in the
  waveform: a plain `np.roll` without any fade is flagged in 88 % of clean seconds.
  It is a discontinuity detector, not a "shift" detector; treat it as such.
* `ClippingDistortion` is drawn with a percentile threshold in 0–20 %; near 0 %
  it leaves the signal untouched, so part of that class is unlearnable by
  construction. The one-line DSP clipping detector beats both CNNs here.
* `TimeStretch` with rate > 1 is zero-padded back to 1 s, so part of that class
  is detected through the silent tail rather than the stretch itself.
* The DSP baselines show which classes do *not* need a neural network: noise,
  band-limiting and clipping are solved by one-line detectors; pitch shift is
  where the CNN earns its keep.