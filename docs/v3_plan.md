# AADD v3 — plan (trained 2026-11-05, weights: https://huggingface.co/gertelrina/aadd)

Merged from three independent design proposals (realism-first, cheap-evolution,
ssl-frame-level); this is the version being executed.

## Goal and positioning

Detect **which** degradation is present and **where** (20 ms frames) in speech,
for a taxonomy that matches real dataset-curation problems (URGENT / ADT style),
with open weights under MIT. Prior work with the same goal (Saishu 2021,
Higham 2022, DAMOS 2026, Kuhlmann 2026) released no weights; v1/v2 cover only
synthetic `audiomentations` transforms on 1-s chunks.

## Taxonomy (12 classes, `Shift` dropped)

| class | generation | parameters | assets (licence) |
|---|---|---|---|
| `noise` | real noise mixed at SNR on active speech; 15 % coloured (white / pink / brown) | SNR −5…30 dB | MUSAN noise (CC BY 4.0) |
| `music` | music bed at SNR | SNR 0…25 dB | MUSAN music (CC BY 4.0 / PD) |
| `reverb` | FFT convolution with a room impulse response, direct path aligned, level matched | simulated + real RIRs; T60 stored | RIRS_NOISES (Apache 2.0); MIT IR survey (CC BY 4.0) for OOD test only |
| `clipping` | hard clip at a percentile of \|x\| or tanh overdrive | 2–30 % samples clipped; drive 2–10× | — |
| `lowpass` | Butterworth low-pass or resample to 4 / 6 / 8 kHz and back | cutoff 1–5 kHz, order 4–8 | — |
| `highpass` | Butterworth high-pass | cutoff 200–1000 Hz, order 2–8 | — |
| `codec` | encode → decode round trip with libsndfile (no ffmpeg) | MP3 / Opus / Vorbis at low bitrates; GSM 06.10, µ-law, A-law, G.721, IMA-ADPCM at 8 kHz (these also set `lowpass`) | libsndfile 1.2.2 in the `soundfile` wheel |
| `packet_loss` | Gilbert–Elliott bursts over 20 ms packets, zero or repeat-last fill | loss 5–30 %, burst 1–5 packets | — |
| `hum` | 50 / 60 Hz + 3–10 harmonics, slow jitter | −35…−15 dB re speech | — |
| `clicks` | Poisson impulses 0.1–3 ms, optional resonance | 2–20 /s | — |
| `pitch_shift` | pitch shift of the interval, crossfaded | ±2…8 semitones | audiomentations / librosa |
| `time_stretch` | time stretch of the interval in place (no zero padding) | rate 0.7–0.9 or 1.15–1.5 | audiomentations / librosa |

Labels are per 20 ms frame. Frames where a degradation is physically
unobservable (clipping that changed no sample, packet loss in silence, pitch /
time edits in silence) and frames within 40 ms of an event boundary are masked
out of the loss.

## Data

| corpus | HF repo | role | licence |
|---|---|---|---|
| LibriSpeech train-clean-100 / dev-clean / test-clean | `Emulated-Inc/enhancement`, local copy | English speech: train / val / test | CC BY 4.0 |
| Russian LibriSpeech (RuLS) | `istupakov/russian_librispeech` | Russian speech: train / val / test | CC BY 4.0 (card), PD source |
| MUSAN (noise, music) | `Emulated-Inc/enhancement` | additive degradations, split 80/10/10 by file | CC BY 4.0 |
| RIRS_NOISES | `Emulated-Inc/enhancement` | simulated RIRs (train/val), real RIRs (test) | Apache 2.0 |
| MIT IR survey | `TaterTotterson/MIT_environmental_impulse_responses` | OOD reverb test | CC BY 4.0 |
| VoiceBank-DEMAND 16k (test) | `JacobLinCool/VoiceBank-DEMAND-16k` | OOD: real recorded noise, clean twins | CC BY 4.0 |
| DNS Challenge 2020 dev-test | `nkdem/DNS-Challenge-2020-DevTest-16k` | OOD: noise, noise + reverb, real recordings | CC BY 4.0 |
| Dynamic-SUPERB reverb / noise detection | `DynamicSuperb/*` | OOD binary reverb / noise tasks | per source |

All repos verified ungated from the training machine on 2026-11-05.

Generation is **on the fly** in data-loader workers (8-s windows, 0–3 events per
window, whole-window or interval 0.5–4 s, random onsets, co-occurrence allowed,
fixed signal-chain order). Validation and test sets are generated once with a
fixed seed and stored. Speech is pre-decoded to int16 memory maps.

## Models

| name | backbone | head | input | ships as |
|---|---|---|---|---|
| `v3` | Whisper-small encoder (88 M, Apache 2.0), fully fine-tuned, 8-s windows (positional table sliced to 400) | weighted sum of 13 hidden states → LayerNorm → 2× Conv1d(k=5) → Linear(12), sigmoid per 20 ms | Whisper log-mel 80 × 800 | fp16 checkpoint, downloaded on first use |
| `v3-lite` | CRNN on 64-band log-mel (≈1 M params) | BiGRU → Linear(12) per 20 ms; trained on the same generator, distilled from `v3` | log-mel 64 × 800 | in the repo, CPU real-time × 100+ |

Training: AdamW, lr 3e-5 (encoder) / 1e-3 (head), warm-up 1 k steps, cosine,
bf16, batch 64 windows, ~20 k steps on one H100. BCE with ignore mask.

## Evaluation

* In-domain (English test-clean, Russian test): per class frame AUC / AP / F1,
  1-s segment F1, event F1 (200 ms collar), clip AUC; per-severity bins.
* Cross-version: `v3` on the v2 test set (class mapping noise / clipping /
  lowpass / highpass / pitch / time), `v2` on the v3 test set for the same
  classes.
* OOD: VoiceBank-DEMAND (noise recall on noisy, false-positive rate on clean
  twins), DNS-2020 (noise, reverb), Dynamic-SUPERB reverb / noise, MIT-IR reverb.
* Baselines: DSP detectors from `aadd/baselines.py` where a class maps.

Success = v3 ≥ v2 on the overlapping classes of the v2 test set, clip AUC ≥ 0.9
on the OOD noise / reverb sets, and false-positive rate on clean OOD speech
reported.

## Out of scope

Low-volume / level classes (Whisper features are max-normalised), wind noise,
bandwidth extension artefacts, TTS / vocoder artefacts, music-only content.
