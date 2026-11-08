# Model card — AADD degradation detectors

Weights are on Hugging Face: [gertelrina/aadd](https://huggingface.co/gertelrina/aadd). v1 and v2 take one second of 16 kHz mono speech and
return 8 sigmoid probabilities in the fixed order
`AddGaussianNoise, PitchShift, TimeStretch, Shift, ClippingDistortion, LowPassFilter, HighPassFilter, None`.
Load them with `aadd.Detector("v1")` / `aadd.Detector("v2")` or `aadd.model.load_checkpoint`.

## Training data (both models)

| | |
|---|---|
| clean source | LibriSpeech **dev-clean** (40 speakers, 2 703 files, 5.4 h) |
| protocol | every file cut into 1-s chunks; each chunk gets, with p = 0.5, one transform drawn uniformly from the 7 classes; chunks re-concatenated; 5 degraded copies + 1 clean per file |
| transforms | `audiomentations`: AddGaussianNoise amp 0.01–0.05 · PitchShift ±8 semitones · TimeStretch rate 0.5–1.5 (length kept: cropped / zero-padded) · Shift ±100 % with rollover · ClippingDistortion percentile 0–20 · LowPassFilter 150–750 Hz · HighPassFilter 2 000–8 000 Hz |
| labels | exactly one label per chunk (a degradation or `None`) |
| size | v1: 16 152 files, ~108 k chunks (not regenerable bit-for-bit: unseeded). v2: regenerated with `aadd make-data --seed 0`: 16 218 files, 108 144 chunks, 10 % of speakers held out for validation |
| test set | LibriSpeech **test-clean**, same protocol, `--seed 1`: 15 720 files, 108 936 chunks, 41.6 % degraded |

## v1 — `aadd_v1_efficientnet_b0_png.pt` (first model, 2025)

| | |
|---|---|
| architecture | torchvision EfficientNet-B0, ImageNet-1k init, classifier replaced by `Linear(1280, 8)` |
| input | mel spectrogram (128 bands, n_fft 2048, hop 512, dB re. chunk max) **rendered by matplotlib** (`librosa.display.specshow`, magma colormap, 4×5 in, tight bbox) → 310×385 RGB PNG → `ToTensor` (no ImageNet normalisation) |
| loss / optimiser | BCEWithLogits, Adam 1e-4, StepLR(5, 0.1), batch 64, 10 epochs planned; the shipped weights are epoch 3 of 10 (0-based epoch 2), selected by bitwise label accuracy |
| trained | 2025-03-11, Colab GPU; original file `efficientnet_b0_sep2_mels_2025-03-11_12-01-06.pth` (46 MB with Adam state), converted without changing weights |
| parameters | 4.06 M |
| feature cost | ≈ 9 ms per second of audio (CPU, matplotlib) |

Results on the regenerated test set (threshold 0.5), from `runs/eval_v1_test_clean.json`:

| class | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| AddGaussianNoise | 1.000 | 1.000 | 1.000 | 1.000 |
| PitchShift | 0.883 | 0.992 | 0.934 | 1.000 |
| TimeStretch | 0.913 | 0.972 | 0.941 | 0.995 |
| Shift | 0.000 | 0.000 | 0.000 | 0.605 |
| ClippingDistortion | 0.837 | 0.893 | 0.864 | 0.973 |
| LowPassFilter | 0.983 | 1.000 | 0.991 | 1.000 |
| HighPassFilter | 0.985 | 1.000 | 0.992 | 1.000 |
| None | 0.896 | 0.957 | 0.926 | 0.918 |

F1 micro 0.908 · macro 0.831 · exact-match 0.906 · file-level Spearman between the
true number of degraded seconds and the predicted count 0.932 (15 720 files).

`None` precision is capped near 0.90 because 96 % of `Shift` chunks, which `v1`
cannot see, are predicted `None`.

## v2 — `aadd_v2_efficientnet_b0_mel.pt`

| | |
|---|---|
| architecture | torchvision EfficientNet-B0, ImageNet-1k init with the RGB stem summed into a 1-channel stem, `Linear(1280, 8)` |
| input | log-mel tensor 1×128×63: n_mels 128, n_fft 1024, hop 256, Hann, power 2, dB re. chunk max clipped at −80 dB, scaled to [0, 1] |
| training | BCEWithLogits, AdamW 5e-4 / wd 1e-2, OneCycle, batch 128, 10 epochs, SpecAugment-style masking (p 0.5), best epoch by speaker-disjoint validation macro-F1 |
| trained | 2026-11-05, Apple M3 Pro (MPS), `aadd train` — see `runs/v2_train_log.json` |
| feature cost | ≈ 1 ms per second of audio |

Results on the regenerated test set (threshold 0.5), from `runs/eval_v2_test_clean.json`;
best epoch 3 of 10 by validation macro-F1 (`runs/v2_train_log.json`):

| class | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| AddGaussianNoise | 0.999 | 1.000 | 0.999 | 1.000 |
| PitchShift | 0.839 | 0.998 | 0.912 | 1.000 |
| TimeStretch | 0.989 | 0.948 | 0.968 | 0.996 |
| Shift | 0.692 | 0.763 | 0.726 | 0.975 |
| ClippingDistortion | 0.784 | 0.856 | 0.818 | 0.971 |
| LowPassFilter | 0.999 | 0.996 | 0.997 | 1.000 |
| HighPassFilter | 0.984 | 1.000 | 0.992 | 1.000 |
| None | 0.960 | 0.925 | 0.942 | 0.982 |

F1 micro 0.930 · macro 0.919 · exact-match 0.927 · file-level Spearman
0.938, MAE of the per-file degradation count 0.40 (v1: 0.908 / 0.831 / 0.906 / 0.932 / 0.55).
Speed on an M3 Pro: 3.09 ms per second of audio on MPS, 39.38 ms on CPU (v1: 14.43 / 132.86).
`Shift` becomes detectable at hop 256 through the waveform seam of the roll (also fires on a
plain `np.roll`): it is effectively a within-second discontinuity detector.

## Intended use and limitations

* Quality gating of speech datasets and spot-checking recordings for the seven
  synthetic degradation types above; per-second localisation.
* **Not** a perceptual quality (MOS) estimator — use NISQA / DNSMOS / SQUIM for that.
* Degradations are synthetic; reverberation, codecs, packet loss, real background
  noise were never seen. Expect `AddGaussianNoise` to fire on broadband noise of
  any kind and `LowPassFilter` on narrow-band audio (telephone), which may or may
  not be what you want.
* One label per second; co-occurring degradations were not in the training data.
* `Shift` is a circular roll of samples inside a second. `v1` cannot see it
  (F1 0); `v2` detects the seam it leaves (F1 0.73) and therefore also fires on
  any other abrupt discontinuity inside a second. Read it as "discontinuity".
* `ClippingDistortion` with a percentile threshold near 0 % is practically the
  identity, so part of that class is unlearnable; `TimeStretch` with rate > 1 is
  partly recognised through the zero-padded tail that `audiomentations` adds.
* English read speech at 16 kHz only. Other languages, music, far-field, and
  sampling rates below 16 kHz were not evaluated (input is resampled to 16 kHz).

## Licence

Code and weights: MIT. Training audio: LibriSpeech (CC BY 4.0).

## v3 — `aadd_v3_whisper_small_frames.pt` ([Hugging Face](https://huggingface.co/gertelrina/aadd))

| | |
|---|---|
| architecture | Whisper-small encoder (Apache 2.0), learnable weighted sum of all 13 hidden states, LayerNorm, two Conv1d (k = 5), linear head |
| output | 12 sigmoids per 20 ms frame: noise, music, reverb, clipping, lowpass, highpass, codec, packet_loss, hum, clicks, pitch_shift, time_stretch |
| input | 16 kHz mono, 8-s windows (Whisper log-mel 80 × 800), sliding with 1-s overlap for longer audio |
| training data | LibriSpeech train-clean-100 + Russian LibriSpeech train; on-the-fly degradations from MUSAN, RIRS_NOISES and DSP / codec generators (see `docs/v3_plan.md`) |
| training | AdamW 3e-5 / 1e-3, cosine, bf16, batch 64, 30 000 steps on one H100 (2.4 h); best validation checkpoint at step 16 000 |
| results | `runs/eval_v3.json`; summary in README → Results and on the Hugging Face model card |
| speed | 23 ms per second of audio on CPU, 6.5 ms on Apple GPU (MPS); Apple MacBook Pro M3 Pro, 36 GB, macOS 15.7, PyTorch 2.15, one process, end to end on 300 LibriSpeech files (`runs/speed_benchmark.json`) |

Limitations specific to v3: noise and hum are the weakest classes (frame AP
0.80–0.89 and 0.77–0.86), partly because the LibriVox source speech itself
contains low-level noise and hum; real-world validation covers noise and
reverberation only; content above 8 kHz is never seen.
