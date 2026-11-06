# AADD in the context of 2013–2026 work on audio degradation detection and non-intrusive quality assessment

Scope: how the AADD task (per-second, multi-type detection of synthetic degradations in speech) relates to
non-intrusive quality metrics, degradation-type detectors, foundation-model approaches, dataset-curation
tooling, classical DSP detectors and real-world evaluation data. AADD's own numbers (v1, v2, DSP baselines)
are not repeated here: see README → Results.

## 1. Where AADD sits

Open non-intrusive speech-quality work since 2019 runs in four stages: compact spectrogram CNNs that regress a
MOS (MOSNet [1], NISQA [2], DNSMOS [3], DNSMOS Pro [4]), SSL-backbone MOS predictors with an explicit
out-of-domain focus (UTMOS/UTMOSv2 [7], SCOREQ [9], WhiSQA [12], Distill-MOS [11]), multi-axis and multi-metric
models (Audiobox Aesthetics [13], Uni-VERSA/ARECHO [21], [22]), and, since 2025, *typed and localised*
assessment that says which degradation occurs and when (LibriAugmented [37], DAMOS [41], Kuhlmann et al.
[38], [39], QualiSpeech [57]). AADD belongs to the last group: a supervised multi-label EfficientNet-B0 on
log-mel chunks that outputs, per second, probabilities for seven audiomentations transforms plus `None`.
The formulation itself is not new: Brandt et al. classified 1-s archive frames in 2017 [26], Reddy et al.
classified VoIP impairments on spectrograms in 2019 [28], Saishu et al. used parallel per-degradation CNNs on
log-mel in 2021 [30], and Higham et al. (Amazon) shipped a 1-s no-reference defect classifier in 2022 [32].
What the 2025–2026 competitors add is a richer taxonomy (real noise, reverb, codecs, packet loss,
co-occurrence), frame-level labels (20–160 ms), SSL backbones and OOD tests; what none of them releases is a
multi-type localiser with open weights. That is the gap AADD occupies with its MIT weights, CLI and reproducible
data protocol, provided its taxonomy and evaluation move towards theirs.

## 2. Non-intrusive quality metrics

| model | year | predicts | input / backbone | code | weights licence | relevance to AADD |
|---|---|---|---|---|---|---|
| NISQA [2] | 2021 | MOS + noisiness, coloration, discontinuity, loudness | 48-band mel (fmax 20 kHz default) → CNN → self-attention → attention pooling | github.com/gabrielmittag/NISQA (MIT) | CC BY-NC-SA 4.0 | `--ms_fmax 8000` needed at 16 kHz; four dimensions map onto AADD classes |
| DNSMOS P.835 [3] / P.808 [100] | 2021–2022 | SIG, BAK, OVRL; P.808 MOS | mel CNN, ONNX, 9.01-s windows hopped by 1 s | microsoft/DNS-Challenge (code MIT) | ONNX files: no explicit licence line | de-facto curation gate; BAK ↔ AddGaussianNoise; cannot localise below 9 s |
| DNSMOS Pro [4] | 2024 | MOS as a Gaussian posterior | compact mel/STFT CNN, 16 kHz, 10 s | github.com/fcumlin/DNSMOSPro | MIT | most apples-to-apples open CNN; its latents separate 16 impairments [37] |
| UTMOS / UTMOSv2 [7] | 2024 | naturalness MOS (synthetic speech) | ImageNet EfficientNetV2 on multi-resolution mel + wav2vec2 branch, fused | sarulab-speech/UTMOSv2, `pip install utmosv2` | MIT | validates "ImageNet CNN on spectrogram" and shows the SSL-fusion upgrade |
| TorchAudio-SQUIM [8] | 2023 | PESQ, STOI, SI-SDR (objective); MOS with non-matching reference (subjective) | ships in `torchaudio.pipelines` | torchaudio | OBJECTIVE CC BY 4.0; SUBJECTIVE CC BY-NC 4.0 | only MOS-family model usable on 1-s chunks; zero extra dependency |
| SCOREQ [9] | 2024 | MOS (NR) or distance to clean (NMR), natural and synthetic variants | wav2vec2 + contrastive regression | `pip install scoreq` | MIT | trained on NISQA_TRAIN_SIM with better OOD behaviour; documents NISQA's cross-domain drop |
| NOMAD [10] | 2024 | perceptual distance to any clean reference | wav2vec2 triplets ordered by ViSQOL on LibriSpeech + 4 degradations × 5 levels | `pip install nomad_audio` | MIT | same data recipe as AADD; its per-type Spearman table is the evaluation protocol AADD needs |
| WhiSQA [12] | 2025 | MOS (+ optional four NISQA dimensions) | frozen whisper-small encoder, weighted layer sum, small transformer head | github.com/leto19/WhiSQA | not stated in survey | stronger NISQA-corpus baseline; cheap SSL branch recipe |
| Distill-MOS [11] | 2025 | MOS | conv-transformer student of an XLS-R teacher | `pip install distillmos` | MIT | CPU-friendly MOS strip for the app; same size class as EfficientNet-B0 |
| Audiobox Aesthetics [13] | 2025 | PQ, PC, CE, CU on 1–10 (speech, music, sound) | WavLM-style encoder + 4 MLP heads, 10-s windows | `pip install audiobox_aesthetics`, JSONL CLI | CC BY 4.0 | PQ ("absence of noise, artefacts, distortion") is the fairest scalar comparator for a degradation count |
| VQScore [17] | 2024 | label-free quality (VQ-VAE quantisation error) | codebook trained on LibriSpeech train-clean-460 | github.com/JasonSWFu/VQscore | not stated in survey | in-domain clean-speech prior; VMC 2024 Track 3 unsupervised baseline |
| SpeechLMScore [15] | 2023 | label-free likelihood under a clean-speech unit LM | HuBERT tokens → LSTM uLM | soumimaiti/speechlmscore_tool | Apache-2.0 | reacts to pitch shift / time stretch, which MOS models ignore |
| SpeechBERTScore [16] | 2024 | reference-aware similarity, length-robust | WavLM-large layer 14 | Takaaki-Saeki/DiscreteSpeechMetrics | MIT | neutral severity label clean-vs-degraded that survives TimeStretch |
| S3QA [18] | 2025 | predicted WavLM cosine distance clean↔degraded | transformer student, degraded audio only | no code | — | principled continuous severity target computable from AADD's own pairs |
| ProxyMOS [19] | 2026 | MOS (multi-teacher distillation) | wav2vec2 student | lab260ru/ProxyMOS, HF weights | not stated in survey | only open MOS model validated on Russian speech (mos260) |
| VoiceMOS 2022–2024 [5], [98], [6]; AudioMOS 2025 [14] | 2022–2025 | challenge series: naturalness MOS, zero-shot OOD, P.835 SIG/BAK/OVRL, aesthetics, multi-rate | SSL backbones + spectrogram branches + ensembles win | BVCC on Zenodo | — | documents OOD failure of all MOS predictors; 16 kHz samples are hard to rank when mixed with wider-band audio |

The quantity each model predicts differs: naturalness of synthetic speech (VoiceMOS line), telecom quality
(NISQA), noise-suppression quality (DNSMOS), objective-metric proxies (SQUIM), or technical production quality
(Audiobox PQ). None predicts a degradation *count*, and only SQUIM works at AADD's 1-s resolution.

### Comparing AADD with MOS predictors

MOS predictors and AADD answer different questions: a MOS is a bounded, perceptual, duration-free score
trained on telecom or TTS conditions, while the AADD count is an unbounded per-file sum of detected
(second, class) pairs. Plotting one against the other, or against an ad-hoc additive "degradation level"
built from the same generator AADD was trained on, is neither like-for-like nor neutral. Pitch shift,
time stretch and very low cut-offs are also outside the training conditions of most MOS models, and
NISQA scores 16 kHz input as band-limited unless `ms_fmax` is set to 8000.

A fair comparison:

1. Run NISQA with `--ms_fmax 8000` (nisqa.tar, all five outputs) and sanity-check that clean test-clean files
   score near the top of the scale before anything else.
2. Compare dimensions, not `mos_pred`: `noi_pred` ↔ AddGaussianNoise, `col_pred` ↔ LowPass/HighPass/Clipping,
   `dis_pred` ↔ TimeStretch/Shift seams, `loud_pred` ↔ no AADD class; the P.804 vocabulary [23] gives the
   definitions.
3. Add the panel practitioners use: DNSMOS BAK/SIG/OVRL/P808 (file level only, 9-s windows), SQUIM
   PESQ/STOI/SI-SDR **per 1-s chunk** (the only model that runs at AADD's native resolution), SCOREQ natural-NR
   and NMR, Distill-MOS, UTMOSv2, Audiobox PQ, VQScore/SpeechLMScore (clean-speech priors that react to pitch and
   time changes). A single VERSA config [74] produces most of them (DNSMOS, NISQA, UTMOSv2, SCOREQ, NOMAD,
   Distill-MOS, Audiobox PQ); SQUIM per 1-s chunk, VQScore and SpeechLMScore need their own calls.
4. Replace AVG with severities that have units: the generator's own parameter per chunk (SNR from the noise
   amplitude or WADA-SNR, clipping percentile, cutoff Hz, semitones, rate) and third-party distances to the clean
   chunk that already exists (PESQ/STOI, SpeechBERTScore for length-changed chunks, WavLM cosine distance as in
   S3QA, NOMAD NMR).
5. Report Spearman **per degradation type** between each metric and the severity (NOMAD Table 1 layout) with
   bootstrap CIs and N, and normalise file-level counts by duration (`degraded_fraction`, which `aadd scan` already
   emits) instead of raw counts.
6. Frame the result as complementarity: MOS says how bad, AADD says what and where. The repo's `aadd eval`
   reports the file-level correlation between the AADD count and the true number of degraded seconds; the panel
   above is the missing external half.

## 3. Degradation type detection and localisation

| work | year | classes / task | granularity | input / backbone | code / weights | relation to AADD |
|---|---|---|---|---|---|---|
| Audio Degradation Toolbox [25] | 2013 | 14 chainable units (noise, clipping, high-pass, aliasing, MP3, reverb, speed-up, wow…) | generator, not detector | Matlab; Python ports audio_degrader [75] | GPL-2+ | canonical taxonomy AADD's audiomentations list should be positioned against |
| Brandt et al. [26] | 2017 | impulsive disturbances in archives | 1-s frames | prewhitened features, supervised classifier | none | earliest 1-s degradation classifier; "detect, then gate the restorer" use case |
| Hennequin et al. (Deezer) [27] | 2017 | lossy-codec traces in PCM | 10-s clips | log-spectrogram CNN; cues = spectral holes, band cuts | none | same cue as LowPass/HighPass; motivates a codec class |
| Reddy et al. [28] | 2019 | noise, reverb, distortion, low volume, clean (VoIP) | 10-s clips | CNN on spectrograms vs engineered features | none | DNSMOS authors' own impairment classifier; industry taxonomy |
| Ragano et al. [29] | 2020 | 4 archive noise types at 0–30 dB SNR | 4-s frames | small CNNs on feature histograms | none | reports rendered-image inputs worse than numeric arrays |
| Saishu et al. [30] | 2021 | additive noise, nonlinear distortion, reverberation | per recording | parallel binary CNNs on log-mel + deltas, Score-CAM | none | closest prior formulation: one sigmoid per degradation on log-mel |
| Hook-Net (Wolff et al.) [31] | 2021 | clicks, MP3 glitches in music | < 10 ms | Wave-U-Net on raw audio, defects inserted on the fly | none | what real localisation looks like; onsets independent of the analysis grid |
| Higham et al. (Amazon) [32] | 2022 | No Defect / Hum / Hiss / Distortion / Clicks | **1-s segments** | AudioSet-pretrained PANN on log-mel | none | same product idea three years earlier; main baseline to cite |
| MT-SLVR [33] | 2023 | predict which augmentations were applied (pitch, stretch, shift, mask, noise) as SSL pretext | clip | contrastive + multi-label augmentation prediction | none | AADD's task as a pretext task; shows time shift is only learnable relative to context |
| URGENT 2024 / 2025 [34], [35] | 2024–2025 | noise, reverb, clipping, bandwidth limitation, codec, packet loss, wind noise; up to 5 at once | simulation scripts | open data-prep repo | open | community taxonomy and simulator that should replace the chunk-and-concatenate generator |
| LibriAugmented / DNSMOS+ [37] | 2025 | 16 impairments on LibriSpeech via audiomentations (real noise, clipping, gain, LP, MP3, pitch, reverb, time mask, stretch, pairs) | 10-s clips | k-NN on MOS-model latents: 94 % top-1 | fcumlin/sqa_latent_classification | near-identical recipe with code; its augmentation list is what AADD lacks |
| Kuhlmann et al., ICASSP 2026 [38] | 2026 | frame-level quality scores, TTS artefact localisation | frames, chunk-based processing | utterance-level SQA model + segment-consistency constraint | — | principled argument for chunked processing (a local distortion must not leak into neighbours) |
| Kuhlmann et al., Odyssey 2026 [39] | 2026 | detection + type of brief degradations, 19 single types + combinations | 50 Hz frames | wav2vec2-large, partial mix-up, supervised contrastive | code hidden during review | direct competitor on "where and which"; OOD on LibriAugmented |
| VoxEffects [40] | 2026 | multi-label effect presence, preset classification, intensity | clip | AudioMAE multi-task baseline | dataset + renderer | "which transforms and how strongly" with parameter-regression heads |
| DAMOS [41] | 2026 | distorted vs not, 20+ URGENT-style distortions injected in 0.5–3 s segments | 160 ms frames | WavLM-Large + boundary-aware localiser feeding a MOS head | none | AADD's dataset design redone at frame level; no release |
| PADAM (Amazon) [42] | 2026 | seven production audio defects | clip | SSL features → contrastive quality head → SVM; synthetic injection incl. encode/transcode | PDF only | industrial confirmation of the use case; argues for codec artefacts |
| SLICE [43] | 2026 | noise detection, T60, distortion intensity as conditioning for enhancement | utterance | multi-task encoder injected into a diffusion enhancer | none | degradation ID as a front-end for universal enhancement; parameter heads |
| Kan & Beigi [44] | 2026 | vinyl defect regimes (clicks, bursts, crackle) | frames + temporal decoding | LPC-residual cues + routed experts, bootstrap CIs | none | frame emissions + smoothing instead of independent chunks; CI reporting |

Two points follow. First, **the exact per-second multi-type formulation exists in prior work and must be
cited**: Higham et al. [32] (1-s segments, five classes including "no defect", log-mel CNN), Brandt et al. [26]
(1-s frames), Saishu et al. [30] (multi-label per-degradation CNNs on log-mel) and Reddy et al. [28].
AADD's actual deltas are the open weights, the pip CLI, the reproducible generator and the per-second UI.

Second, the field has moved from clip tags to **localisation at 20–160 ms with onsets independent of the
analysis grid** ([31], [38], [39], [41], [44]) and to **severity alongside type** (T60 in [47] and [43],
intensity in [40], [43]). AADD's degradations are generated on the same 1-s grid the classifier uses, so
chunk-level detection never faces boundary misalignment, and only one degradation is applied per chunk, so the
multi-label head is never trained or tested on co-occurrence ([33] shows the co-occurring formulation). Standard
taxonomies between them cover reverberation (ADT [25], URGENT [35], P.804 [23]; blind RT60 estimation is itself a
mature line, [47]), codecs (ADT, URGENT) and packet loss / interruptions (URGENT, P.804 discontinuity, P.563 mutes
[24]); AADD has none of these and includes Shift, which is not an acoustic degradation.

Input representation. No surveyed degradation detector renders spectrograms through a plotting library; all
feed log-mel, gammatone or STFT arrays. McLoughlin et al. state there is "no logical justification for processing
using colour spectrograms" (tool-dependent mapping, about 3× front-end cost) [45]; Palanisamy et al. show
ImageNet CNNs work on channel-replicated log-mel arrays [46]; Ragano et al. report rendered images performing
worse than numeric arrays [29]. AADD v1 (matplotlib PNG, magma colormap, 310×385 RGB) is reproduced pixel-for-pixel in
`aadd/features.py`; v2 feeds the 1×128×63 log-mel tensor to the same backbone, which is the standard path.

## 4. Foundation models and audio LLMs for quality

| family | representative work | evidence relevant to AADD |
|---|---|---|
| SSL encoder + small head | SSL-MOS baseline of VoiceMOS 2024 [6]; Oliveira et al. [51]; Tamm et al. [52]; WhiSQA [12]; DAMOS [41]; BAM [53] | Whisper-small/base and WavLM-Base+ give almost all of the quality signal of XLS-R-2B [51]; noise and room-acoustics information lives in early layers, so a learnable layer-weighted sum beats the last layer [52]; frame-level (20–160 ms) heads over whole files are how localisation is done [53], [41] |
| CNN on mel, audio-pretrained | EfficientAT mn10_as / DyMN [49] | MobileNetV3 pretrained on ImageNet then distilled from PaSST on AudioSet; mn10_as 4.88 M params, 128-mel tensor computed inside the model, MIT; same budget as EfficientNet-B0 |
| spectrogram transformers | BEATs / PaSST [50] | best frozen general-audio embeddings (~90 M params) but no speech priors; 15–20× the cost of mn10 |
| CLAP zero-shot | PAM [54]; Martel et al. [55] | scalar "clean vs artefacts" only, blind to reverb/colouration, no type; CLAP linearly encodes RT60 and loudness, but amplitude-invariant backbones discard the level cue that AADD's noise and clipping classes rely on |
| audio LLMs as judges | SpeechLLM-as-Judges [58]; AudioJudge [59]; MRMAD [60] | zero-shot 7B audio LLMs: PCC 0.073 with overall quality (Qwen2-Audio-7B); fixing it costs ~43 A100-hours of LoRA; GPT-4o-audio reaches roughly human-level pairwise agreement but stays below a fine-tuned small model and gives no per-second output; 18 audio LLMs fail degradation-type identification on MRMAD |
| descriptive quality | ALLD [56]; QualiSpeech [57]; XSQ-AST [61] | the field is moving from a scalar MOS to typed, temporally located descriptions ("noise at 0–3 s"), i.e. AADD's output format; saliency over time from an AST is an alternative to frame labels |
| multi-metric | Uni-VERSA [22]; ARECHO [21]; ANCHOR [96] | one SSL network predicts many metrics; ANCHOR [96] studies chunk-level quality and finds a 4–6 s perceptual context horizon, an argument that 1-s chunks are too short for MOS but fine for type detection |

Recommended next backbone. (1) Keep the mel-tensor path of v2 and swap EfficientNet-B0 for EfficientAT
`mn10_as` [49]: same parameter class, pretrained on both ImageNet and AudioSet, takes the 128-mel tensor directly,
fine-tunes in minutes per epoch on an M-series GPU; this is the cheapest, highest-certainty gain. (2) Add a frozen
whisper-small encoder or WavLM-Base+ branch with a softmax-weighted layer sum and late fusion, the UTMOSv2 [7] /
WhiSQA [12] recipe; features are precomputed once and the head trains on CPU. (3) With a GPU, fine-tune
WavLM-Base+ end-to-end with 20-ms frame-level 8-way sigmoid heads pooled to 1 s (BAM/DAMOS style, [53], [41]),
which removes the chunk-cutting artefacts and makes a time shift detectable as a boundary event. Check that the
chosen backbone keeps level information or feed RMS as a side feature [55]. Audio LLMs and CLAP stay as reported
negative baselines, not as the model.

## 5. Dataset-curation tooling and interchange formats

| tool / pipeline | year | quality signal used | output format | licence | gap relevant to AADD |
|---|---|---|---|---|---|
| Emilia-Pipe [62] | 2024 | DNSMOS P.835 OVRL > 3.0 gate (+ LID, char-rate IQR) | one JSON per input, per-segment `{text,start,end,language,dnsmos}` | code MIT; data CC BY-NC 4.0 | scalar MOS gate; an `aadd` object fits next to `dnsmos` |
| AutoPrep [63] | 2024 | DNSMOS/PDNSMOS filtering stage | — (closed models) | — | second canonical in-the-wild pipeline with a quality-filter slot |
| WenetSpeech4TTS [64]; WenetSpeech-Yue [65] | 2024–2025 | DNSMOS P.808 tiers (> 4.0 / 3.8 / 3.6); DNSMOS > 2.5 and Brouhaha SNR > 25 dB | per-utterance quality columns | — | production corpora already ship per-utterance quality fields; type-of-degradation column is the missing axis |
| Data-Speech / Parler-TTS [66] | 2024 | Brouhaha SNR and C50, SQUIM PESQ/SI-SDR, binned into text tags | dataset columns | MIT | "degradation as per-utterance column" design AADD extends |
| NeMo SDP [67] | 2024 | `TorchSquimObjectiveQualityMetricsProcessor`, `BandwidthEstimationProcessor`, `PreserveByValue`, `LambdaExpression` | JSONL manifest `{audio_filepath,duration,text,…}` with `segments[i].metrics` | Apache-2.0 | no noise/reverb/clipping processor |
| NeMo SDE [68] | 2021 | peak level, bandwidth; any extra manifest field becomes a sortable column | JSONL manifest | Apache-2.0 | the `aadd report` dashboard re-implements a subset |
| NeMo Curator [69]; Granary [70] | 2025 | WER/CER/duration only | JSONL | Apache-2.0 | no acoustic-quality stage even at 643k-hour scale |
| lhotse / Libriheavy [71] | 2024 | — | `cut.custom` dict, `attach_tensor` per-frame arrays | Apache-2.0 | second integration target; Libriheavy is a large real-defect demo set |
| Brouhaha [72] | 2023 | per-16-ms VAD, SNR (dB), C50 | per-frame `.npy` + per-file mean txt | code MIT; HF model OpenRAIL, gated | nearest tool in spirit; AADD has no reverb class and no SNR regression |
| Whilter / AITW [73] | 2025 | multi-speaker, music, foreign language, noise, synthetic (per utterance) | — | dataset CC BY 4.0; weights not released | noise is the hard class on real data (F1 63.2); AITW is a real-world noise test set |
| VERSA [74] | 2025 | 90+ metrics incl. DNSMOS, NISQA, UTMOS, SCOREQ, NOMAD, Distill-MOS, Audiobox | YAML-driven, Kaldi ARK/SCP, Slurm | MIT | registering AADD as an independent metric puts it next to every baseline |
| audio_degrader [75]; ADT [25] | 2013 / 2021 | generators (mix at SNR, gain, compression, resample, IR convolution, MP3) | CLI recipe string | GPL-2+ (ADT) / GPL-3 (audio_degrader) | realistic replacement for audiomentations ops; recipe string logs ground truth per chunk |
| Di Bernardo et al. [76] | 2025 | 24 pipeline configs: NISQA / DNSMOS thresholds, SQUIM, WADA-SNR, T30/C50 | composite objective | code open | ready protocol to test "does an AADD gate help"; notes no unified threshold criterion |

Observation: no pip-installable tool does per-second typed degradation detection. Brouhaha gives SNR and C50 per
frame, SDP has SQUIM and bandwidth, Whilter has no released weights, and a GitHub search for "audio degradation
detection" returns essentially only `gertelrina/aadd` [78]. Every surveyed pipeline gates on a single MOS with an
ad-hoc threshold between 2.5 and 4.0 and none outputs the type of degradation.

What AADD's output should look like. Today `aadd scan` writes a per-file CSV (`path, duration_s, n_segments,
degraded_segments, degraded_fraction, aadd_count, detected, status, error, p_<class>`), a per-second CSV (`path,
start_s, end_s, labels, p_<class>`) and a JSON with `file` and `segments`. To plug into the tools above it should
additionally emit a NeMo-compatible JSONL manifest, one line per file:

```json
{"audio_filepath": "a.flac", "duration": 4.0, "aadd_model": "v2", "aadd_segment_sec": 1.0,
 "aadd_threshold": 0.5, "aadd_degraded_fraction": 0.5, "aadd_count": 2,
 "aadd_flags": ["ClippingDistortion", "HighPassFilter"], "aadd_max_p_ClippingDistortion": 1.0,
 "segments": [{"start": 0.0, "end": 1.0, "metrics": {"aadd_p_AddGaussianNoise": 0.01, "...": "..."}}]}
```

Flat top-level `aadd_*` keys make SDE display and sort them with no code; `segments[i].metrics` matches the
SQUIM processor convention so a `PreserveByValue: {input_value_key: aadd_degraded_fraction, operator: le}`
filter works; the same object can be written as `cut.custom["aadd"]` plus `attach_tensor("aadd_probs", N×8,
frame_shift=1.0)` for lhotse, as an `"aadd"` block next to `"dnsmos"` for Emilia, and as a VERSA metric entry.
Two one-file contributions (an `AADDDegradationProcessor` subclassing SDP's `BaseParallelProcessor`, a
`- name: aadd` metric in VERSA) make AADD usable by the NeMo and ESPnet communities without its HTML report.
Audiobox Aesthetics' `audio-aes input.jsonl > output.jsonl` contract (`{path,start_time,end_time}` in, one key per
axis out) is the cleanest CLI template to copy.

## 6. Classical DSP detectors per AADD class

| AADD class | one-line detector | references | in `aadd/baselines.py` | needs a neural net? |
|---|---|---|---|---|
| ClippingDistortion | fraction of samples sitting exactly on the chunk's min/max, or an amplitude-histogram tail peak at the clip level; run-length of flat tops | Laguna & Lerch [80]; Aleinik & Matveev R_cl [81]; Eaton & Naylor ILAH for coded speech [82]; ClipDaT / `pip install clipdetect` [83]; Essentia SaturationDetector [94] | yes (fraction of samples at the chunk min/max) | no; hard clipping at a percentile is located by thresholding by construction; draws near 0 % are practically clean |
| LowPassFilter | effective bandwidth f_max = highest frequency within −50 dB of the spectral peak; or share of energy below 1 kHz | Bakhturina et al. [84] (NeMo SDP `BandwidthEstimationProcessor`, n_fft 512, −50 dB, stride 0.01 s); Fuhs et al. [85]; URGENT audit [36] | yes (low-band energy share) | no; narrowband detection was a solved two-class problem in 2009 |
| HighPassFilter | share of energy above 1 kHz (low-band ratio < 0.05) | same as above | yes | no |
| AddGaussianNoise | WADA-SNR from the amplitude distribution (Gamma speech + Gaussian noise); minimum-statistics noise floor; 10th-percentile 4–8 kHz frame energy; NIST STNR | Kim & Stern [86] (NumPy gist, urgent2024_analysis); Martin [87]; snreval [88]; Essentia SNR [94] | yes (4–8 kHz noise floor, dB) | no for white Gaussian noise at fixed amplitude; real noise at low SNR is where learned models matter [29], [73] |
| PitchShift | MFCC statistical moments + SVM (pitch shift = stretch + resample scales the spectral envelope); down-shifts leave a brick-wall band limit at 8000·2^(n/12) Hz from librosa's resampling; median F0 alone is useless (shifted F0 stays in range) | Wu, Wang & Huang [90]; Ye et al. for weak shifts < 4 st [48]; Laroche & Dolson phasiness [97]; resampling forensics (second-derivative periodicity; Zhou et al. 2024 [99], abstract not verified by the survey) | no (constant score) | yes for weak shifts; this is the class where the CNN earns its keep; report F1 binned by abs(semitones) |
| TimeStretch | rate > 1: trailing zeros from `leave_length_unchanged=True` (a label artefact, not a stretch cue); rate < 1: vertical phase-coherence ("phasiness") of the phase vocoder; syllable-rate estimators are too coarse at 1 s | Laroche & Dolson [89], [97]; Wang & Narayanan [91] | yes (trailing-zero fraction; rate > 1 half only) | yes for the slow-down half; fix the generator so the zero tail cannot leak |
| Shift | LPC-residual discontinuity detector (abs(e(t)) > μ·σ), Essentia Click/DiscontinuityDetector | Godsill & Rayner [92]; Muehlbauer [93]; Essentia [94]; Kan & Beigi [44] | yes (largest sample jump / RMS) | not a model question: rollover with a 5-ms fade leaves no evidence inside the chunk; redefine (rollover=False, or fade 0) or drop |

Three further non-deep comparators: openSMILE eGeMAPSv02 functionals + logistic regression as a classical-ML
baseline on the same chunks [95]; ITU-T P.563 [24], the single-ended standard that first detects a dominant
distortion class (noise, mutes, robotisation) and only then maps to MOS, i.e. AADD's "type first, then severity"
order from 2004; and the Essentia audio-problem suite, which Ozturk ran over all 51k FSD50K clips for dataset-level
statistics [94], the same deliverable as the `aadd report` dashboard. The URGENT 2024 audit [36] did the same at
corpus scale with WADA-SNR and the −50 dB bandwidth rule and found declared ≠ effective bandwidth in most of
DNS5-LibriVox and CommonVoice. Thresholds for the rules above are deliberately not baked in; `aadd eval
--baselines` reports threshold-free ROC-AUC/AP and best achievable F1 (see README → Results). The augmentation
backend matters for every pitch/stretch cue: audiomentations 0.38 used librosa's phase vocoder, later versions
default to signalsmith_stretch with different artefacts; `aadd make-data --stretch-method` pins it.

## 7. Real-world evaluation data with degradation ground truth

| dataset | ground truth | what to test | access |
|---|---|---|---|
| NISQA_TRAIN_SIM / VAL_SIM (NISQA Corpus) [2] | 10k / 2.5k clips, 1,258 simulated conditions from 9 base distortions (codecs, packet loss, bandpass, clipping, Gaussian/MNRU noise, DNS/AudioSet/DEMAND noise), per-file condition metadata, MOS + four dimensions | map bandpass → LowPass/HighPass, clipping → ClippingDistortion, noise → AddGaussianNoise, codec → "other"; first OOD number for the shipped checkpoints; NISQA_VAL_LIVE for real recordings | Zenodo 4728081, non-commercial research use |
| URGENT 2024 / 2025 [34], [35], [36] | 7 distortion types with open simulation scripts, up to 5 co-occurring, multiple sampling rates and languages, real recordings | taxonomy transfer (reverb, codec, packet loss absent from AADD), co-occurrence, sampling-rate robustness; URGENT 2026 Track 2 (quality of enhanced speech) is a venue where a type detector is directly useful | open |
| DNS-Challenge data [79] | noise (AudioSet CC BY 4.0, Freesound CC0, DEMAND CC BY-SA 3.0) and OpenSLR RIRs (Apache 2.0), ~1 TB; VoiceMOS 2024 Track 3 [6] adds P.835 SIG/BAK/OVRL labels on noisy/enhanced speech | real-noise version of AddGaussianNoise; reverb class if added; BAK as the external referee for the noise class | download scripts in the repo |
| AITW (Whilter) [73] | 21,414 human-annotated clips (~64 h) from Emilia/YODAS: speaker count, non-English, background music, noisy/poor quality, synthetic | AADD's noise / any-degradation output vs the human "noisy" label, next to Whilter's noise F1 63.2 | Zenodo, CC BY 4.0 |
| QualiSpeech [57] | 12,450 recordings (synthetic, NISQA simulated and live, GigaSpeech) with noise/distortion type and temporal-location descriptions | precision of AADD detections against human descriptions; template for an NL summary | Hugging Face |
| VoiceBank-DEMAND, LibriMix [77] | noise type and SNR (0/5/10/15 dB; WHAM! mixtures) | known-SNR curve for the noise class; overlapped speech as a pseudo "multi-speaker" class | open |
| MOS-Bench [20] | 17 MOS test sets (BVCC, SOMOS, SingMOS, TMHINT-QI, Tencent, NISQA, …) | correlate `degraded_fraction` with human MOS on real data | SHEET toolkit |
| mos260 (ProxyMOS) [19] | 4,600 Russian TTS utterances, 38 conditions, MOS | sanity check on Russian audio if the tool is used beyond LibriSpeech English | Hugging Face |
| LibriAugmented [37], DAMOS set [41], Kuhlmann partial mix-up [39], VoxEffects [40] | 16 impairments / frame-level distortion labels / frame-level mix-up targets / effect presets and intensity | closest synthetic benchmarks; the DAMOS set and the Kuhlmann partial-mix-up data are not released; LibriAugmented's generation code is on GitHub (fcumlin/sqa_latent_classification) but the surveys disagree on whether the rendered clips are downloadable; VoxEffects ships a renderer | DAMOS / Kuhlmann: not released; LibriAugmented: code public, rendered clips unclear (check before relying on it); VoxEffects: renderer |
| MRMAD [60] | degradation type identification, severity comparison and ranking across speech, music, sound (audio-LLM benchmark) | compare a specialised detector against 18 audio LLMs on the type-identification task | github.com/Bose/MRMAD |

The obvious next evaluation is NISQA_VAL_SIM with its condition metadata, because it needs no retraining and
gives the first number on real speech for four of AADD's classes; URGENT validation data then covers the classes
AADD lacks, and AITW gives a human-labelled noise test. The DAMOS set and the Kuhlmann partial-mix-up data are
not released; LibriAugmented's generation code is on GitHub (fcumlin/sqa_latent_classification) but the surveys
disagree on whether the rendered clips are downloadable – check before relying on it. Releasing AADD's own
generator config, seeds and per-chunk CSVs (LibriSpeech is CC BY 4.0) as a benchmark is therefore itself a
citable contribution [77].

## 8. References

1. Lo et al., MOSNet: Deep Learning-based Objective Assessment for Voice Conversion, 2019. https://arxiv.org/abs/1904.08352
2. Mittag, Naderi, Chehadi, Möller, NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets, 2021. https://arxiv.org/abs/2104.09494 (code https://github.com/gabrielmittag/NISQA, corpus https://zenodo.org/records/4728081)
3. Reddy, Gopal, Cutler, DNSMOS P.835: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors, 2022. https://arxiv.org/abs/2110.01763
4. Cumlin et al., DNSMOS Pro: A Reduced-Size DNN for Probabilistic MOS of Speech, 2024. https://www.isca-archive.org/interspeech_2024/cumlin24_interspeech.html
5. Huang et al., The VoiceMOS Challenge 2022, 2022. https://arxiv.org/abs/2203.11389
6. Huang et al., The VoiceMOS Challenge 2024: Beyond Speech Quality Prediction, 2024. https://arxiv.org/abs/2409.07001
7. Baba, Nakata, Saito, Saruwatari, The T05 System for the VoiceMOS Challenge 2024 (UTMOSv2), 2024. https://arxiv.org/abs/2409.09305
8. Kumar et al., TorchAudio-Squim: Reference-less Speech Quality and Intelligibility Measures in TorchAudio, 2023. https://arxiv.org/abs/2304.01448
9. Ragano, Skoglund, Hines, SCOREQ: Speech Quality Assessment with Contrastive Regression, 2024. https://arxiv.org/abs/2410.06675
10. Ragano, Skoglund, Hines, NOMAD: Unsupervised Learning of Perceptual Embeddings for Speech Enhancement and Non-matching Reference Audio Quality Assessment, 2024. https://arxiv.org/abs/2309.16284
11. Stahl, Gamper, Distillation and Pruning for Scalable Self-Supervised Representation-Based Speech Quality Assessment (Distill-MOS), 2025. https://arxiv.org/abs/2502.05356
12. Close, Hong, Hain, Goetze, WhiSQA: Non-Intrusive Speech Quality Prediction Using Whisper Encoder Features, 2025. https://arxiv.org/abs/2508.02210
13. Tjandra et al., Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound, 2025. https://arxiv.org/abs/2502.05139
14. Huang et al., The AudioMOS Challenge 2025, 2025. https://arxiv.org/abs/2509.01336
15. Maiti, Peng, Saeki, Watanabe, SpeechLMScore: Evaluating Speech Generation Using Speech Language Model, 2023. https://arxiv.org/abs/2212.04559
16. Saeki et al., SpeechBERTScore: Reference-Aware Automatic Evaluation of Speech Generation Leveraging NLP Evaluation Metrics, 2024. https://arxiv.org/abs/2401.16812
17. Fu, Hung, Tsao, Wang, Self-Supervised Speech Quality Estimation and Enhancement Using Only Clean Speech (VQScore), 2024. https://arxiv.org/abs/2402.16321
18. Ogg, Bishop, Yi, Robinson, Self-Supervised Speech Quality Assessment (S3QA), 2025. https://arxiv.org/abs/2506.01655
19. Trokunov, Borodin, Vasiliev, Mkrtchian, ProxyMOS: Label-Free Speech Quality Assessment by Multi-Teacher Distillation with Adaptive Routing, 2026. https://arxiv.org/abs/2610.00419
20. Huang, Cooper, Toda, MOS-Bench: Benchmarking Generalization Abilities of Subjective Speech Quality Assessment Models; SHEET toolkit, 2024. https://arxiv.org/abs/2411.03715
21. Shi et al., ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation, 2025. https://arxiv.org/abs/2505.24518
22. Shi, Shim, Watanabe, Uni-VERSA: Versatile Speech Assessment with a Unified Network, 2025. https://arxiv.org/abs/2505.20741
23. Naderi, Cutler, Ristea, Multi-dimensional Speech Quality Assessment in Crowdsourcing (ITU-T P.804 dimensions), 2024. https://arxiv.org/abs/2309.07385
24. ITU-T Recommendation P.563: Single-ended method for objective speech quality assessment in narrow-band telephony applications, 2004. https://www.itu.int/rec/T-REC-P.563
25. Mauch, Ewert, The Audio Degradation Toolbox and Its Application to Robustness Evaluation, 2013. https://archives.ismir.net/ismir2013/paper/000145.pdf
26. Brandt, Doclo, Gerkmann, Bitzer, Impulsive Disturbances in Audio Archives: Signal Classification for Automatic Restoration, 2017. https://doi.org/10.17743/jaes.2017.0032
27. Hennequin, Royo-Letelier, Moussallam, Codec Independent Lossy Audio Compression Detection, 2017. https://research.deezer.com/publication/2017/04/05/icassp-hennequin.html
28. Reddy, Cutler, Gehrke, Supervised Classifiers for Audio Impairments with Noisy Labels, 2019. https://arxiv.org/abs/1907.01742
29. Ragano, Benetos, Hines, Audio Impairment Recognition using a Correlation-Based Feature Representation, 2020. https://arxiv.org/abs/2003.09889
30. Saishu, Poorjam, Christensen, A CNN-based approach to identification of degradations in speech signals, 2021. https://link.springer.com/article/10.1186/s13636-021-00198-4
31. Wolff, Mignot, Roebel, Audio Defect Detection in Music with Deep Networks (Hook-Net), 2021. https://arxiv.org/abs/2202.05718
32. Higham, Bagla, Haralampieva, A No-Reference Model for Detecting Audio Artifacts using Pretrained Audio Neural Networks, 2022. https://openaccess.thecvf.com/content/WACV2022W/VAQ/papers/Higham_A_No-Reference_Model_for_Detecting_Audio_Artifacts_Using_Pretrained_Audio_WACVW_2022_paper.pdf
33. Heggan, Hospedales, Budgett, Yaghoobi, MT-SLVR: Multi-Task Self-Supervised Learning for Transformation In(Variant) Representations, 2023. https://arxiv.org/abs/2305.17191
34. Zhang et al., URGENT Challenge: Universality, Robustness, and Generalizability for Speech Enhancement, 2024. https://arxiv.org/abs/2406.04660
35. Saijo et al., Interspeech 2025 URGENT Speech Enhancement Challenge, 2025. https://arxiv.org/abs/2505.23212
36. Zhang et al., Lessons Learned from the URGENT 2024 Speech Enhancement Challenge, 2025. https://arxiv.org/abs/2506.01611
37. Cumlin et al., Impairments are Clustered in Latents of Deep Neural Network-based Speech Quality Models (LibriAugmented, DNSMOS+), 2025. https://arxiv.org/abs/2504.21528
38. Kuhlmann, Werning, von Neumann, Haeb-Umbach, Speech Quality-Based Localization of Low-Quality Speech and Text-to-Speech Synthesis Artefacts, 2026. https://arxiv.org/abs/2601.21886
39. Kuhlmann, Cord-Landwehr, Haeb-Umbach, Speech Quality Embeddings for Improved Detection and Classification of Degradations in Speech Signals, 2026. https://arxiv.org/abs/2605.21332
40. Zhang, Özer, Yamagishi, VoxEffects: A Speech-Oriented Audio Effects Dataset and Benchmark, 2026. https://arxiv.org/abs/2604.12389
41. Li, Dong, Yan, DAMOS: Learning Distortion-Aware Speech Quality Assessment through Explicit Distortion Localization, 2026. https://arxiv.org/abs/2608.21176
42. Mackin et al., PADAM: Perceptual Audio Defect Assessment Model, 2026. https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=4172
43. Moon, Jung, Choo, SLICE: Speech Enhancement via Layer-wise Injection of Conditioning Embeddings, 2026. https://arxiv.org/abs/2603.05302
44. Kan, Beigi, Signal-Informed Temporal Routing for Vinyl Defect Regime Detection (arXiv preprint), 2026. https://arxiv.org/abs/2609.23164
45. McLoughlin et al., Spectrogram features for audio and speech analysis, 2026. https://arxiv.org/abs/2603.14917
46. Palanisamy, Singhania, Yao, Rethinking CNN Models for Audio Classification, 2020. https://arxiv.org/abs/2007.11154
47. Gamper, Tashev, Blind Reverberation Time Estimation Using a Convolutional Neural Network, IWAENC 2018. https://doi.org/10.1109/IWAENC.2018.8521241 (also https://www.semanticscholar.org/paper/7df1e0b29be73961cdd65d776d33adbfd8032a65)
48. Ye, Lao, Yan, Wang, Identification of Weakly Pitch-Shifted Voice Based on Convolutional Neural Network, 2020. https://onlinelibrary.wiley.com/doi/10.1155/2020/8927031
49. Schmid, Koutini, Widmer, Efficient Large-Scale Audio Tagging via Transformer-to-CNN Knowledge Distillation (EfficientAT), 2023. https://arxiv.org/abs/2211.04772
50. Chen et al., BEATs: Audio Pre-Training with Acoustic Tokenizers (and PaSST), 2022. https://arxiv.org/abs/2212.09058
51. Oliveira et al., Evaluation of Speech Representations for MOS prediction, 2023. https://arxiv.org/abs/2306.09979
52. Tamm, Vandenberghe, Van hamme, Analysis of XLS-R for Speech Quality Assessment, 2023. https://arxiv.org/abs/2308.12077
53. Zhong, Li, Yi, Enhancing Partially Spoofed Audio Localization with Boundary-aware Attention Mechanism (BAM), 2024. https://arxiv.org/abs/2407.21611
54. Deshmukh et al., PAM: Prompting Audio-Language Models for Audio Quality Assessment, 2024. https://arxiv.org/abs/2402.00282
55. Martel, Hennessy-Priest, Cho, Probing Low-Level Acoustic Attribute Encoding in CLAP Audio Embeddings, 2026. https://arxiv.org/abs/2607.03806
56. Chen et al., Audio Large Language Models Can Be Descriptive Speech Quality Evaluators, 2025. https://arxiv.org/abs/2501.17202
57. Wang et al., QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions, 2025. https://arxiv.org/abs/2503.20290
58. Wang et al., SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation, 2025. https://arxiv.org/abs/2510.14664
59. Manakul, Gan, Ryan, Khan, Sirichotedumrong, Pipatanakul, Held, Yang, AudioJudge: Understanding What Works in Large Audio Model Based Speech Evaluation, 2025. https://arxiv.org/abs/2507.12705
60. Li et al., MRMAD: A Multi-Round Multi-Audio Benchmark for Evaluating Acoustic Degradation Perception in Large Audio-Language Models, 2026. https://arxiv.org/abs/2608.22236
61. Heritage et al., XSQ-AST: An Explainable Audio Spectrogram Transformer Framework for Localising Synthetic Speech Artifacts, 2026. https://arxiv.org/abs/2609.24770
62. He et al., Emilia / Emilia-Pipe (Amphion preprocessors), 2024. https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia
63. Yu et al., AutoPrep: An Automatic Preprocessing Framework for In-the-Wild Speech Data, 2024. https://arxiv.org/abs/2309.13905
64. Ma et al., WenetSpeech4TTS: A 12,800-hour Mandarin TTS Corpus for Large Speech Generation Model Benchmark, 2024. https://arxiv.org/abs/2406.05763
65. Li et al., WenetSpeech-Yue: A Large-scale Cantonese Speech Corpus with Multi-dimensional Annotation (WenetSpeech-Pipe), 2025. https://arxiv.org/abs/2509.03959
66. Lyth, King, Natural language guidance of high-fidelity text-to-speech with synthetic annotations (Data-Speech), 2024. https://arxiv.org/abs/2402.01912
67. NVIDIA, NeMo Speech Data Processor (SDP), 2024. https://github.com/NVIDIA/NeMo-speech-data-processor
68. Bakhturina et al., NeMo Speech Data Explorer (SDE), 2021. https://github.com/NVIDIA-NeMo/Speech/tree/stable/tools/speech_data_explorer
69. NVIDIA, NeMo Curator – audio curation stages, 2025. https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-assessment/wer-filtering
70. Koluguri et al., Granary: Speech Recognition and Translation Dataset in 25 European Languages, 2025. https://arxiv.org/abs/2505.13404
71. Kang et al., Libriheavy: a 50,000 hours ASR corpus (+ lhotse manifests), 2024. https://github.com/k2-fsa/libriheavy (lhotse: https://github.com/lhotse-speech/lhotse)
72. Lavechin et al., Brouhaha: multi-task training for voice activity detection, speech-to-noise ratio, and C50 room acoustics estimation, 2023. https://github.com/marianne-m/brouhaha-vad
73. Ravenscroft et al., Whilter: A Whisper-based Data Filter for "In-the-Wild" Speech Corpora (+ AITW dataset), 2025. https://arxiv.org/abs/2507.21642
74. Shi et al., VERSA: A Versatile Evaluation Toolkit for Speech, Audio, and Music, 2025. https://arxiv.org/abs/2412.17667
75. Molina, audio_degrader: audio degradation toolbox in Python with CLI, 2021. https://github.com/emilio-molina/audio_degrader
76. Di Bernardo et al., Evaluation of preprocessing pipelines in the creation of in-the-wild TTS datasets, 2025. https://arxiv.org/abs/2510.03111
77. Cosentino et al., LibriMix: An Open-Source Dataset for Generalizable Speech Separation (and VCTK-DEMAND / noisy LibriSpeech variants), 2020. https://github.com/JorisCos/LibriMix
78. audiocurate: CLI to curate and quality-filter audio datasets into reproducible manifests, 2026. https://github.com/ella-brown629/audiocurate
79. Microsoft, DNS-Challenge repository (dnsmos_local.py, ONNX models, DNS datasets), 2022. https://github.com/microsoft/DNS-Challenge
80. Laguna, Lerch, An Efficient Algorithm for Clipping Detection and Declipping Audio, 2016. https://musicinformatics.gatech.edu/wp-content_nondefault/uploads/2016/09/Laguna_Lerch_2016_An-Efficient-Algorithm-For-Clipping-Detection-And-Declipping-Audio.pdf
81. Aleinik, Matveev, Detection of Clipped Fragments in Speech Signals, 2014. https://www.researchgate.net/publication/260436661_Detection_of_Clipped_Fragments_in_Speech_Signals
82. Eaton, Naylor, Detection of Clipping in Coded Speech Signals, 2013. https://zenodo.org/records/43395 (also https://www.semanticscholar.org/paper/Detection-of-clipping-in-coded-speech-signals-Eaton-Naylor/374f3de0889cf3be2b37b543ae6e2b07ae2df321)
83. Hansen, Stauffer, Xia, Nonlinear waveform distortion: Assessment and detection of clipping on speech data and systems (ClipDaT), 2021. https://www.sciencedirect.com/science/article/pii/S0167639321000832
84. Bakhturina, Lavrukhin, Ginsburg, Zhang, Hi-Fi Multi-Speaker English TTS Dataset (bandwidth estimation rule), 2021. https://arxiv.org/abs/2104.01497
85. Fuhs, Jin, Schultz, Detecting bandlimited audio in broadcast television shows, 2009. https://www.researchgate.net/publication/220731826_Detecting_bandlimited_audio_in_broadcast_television_shows
86. Kim, Stern, Robust signal-to-noise ratio estimation based on waveform amplitude distribution analysis (WADA-SNR), 2008. https://www.isca-archive.org/interspeech_2008/kim08e_interspeech.pdf (NumPy port: https://gist.github.com/johnmeade/d8d2c67b87cda95cd253f55c21387e75)
87. Martin, Noise power spectral density estimation based on optimal smoothing and minimum statistics, 2001. https://www.csd.uoc.gr/~hy578/2005/projects/ieee_sp_tsap_2001009_05jul_0504mart.pdf
88. NIST STNR / LabROSA snreval toolkit, 2010. https://github.com/ludlows/snreval
89. Laroche, Dolson, Improved phase vocoder time-scale modification of audio, IEEE Trans. Speech and Audio Processing 7(3):323–332, 1999. https://doi.org/10.1109/89.759041
90. Wu, Wang, Huang, Identification of Electronic Disguised Voices, 2014. https://www.semanticscholar.org/paper/Identification-of-Electronic-Disguised-Voices-Wu-Wang/8758fa3bd4fe67addd19200ad2a59f037dbf78fb
91. Wang, Narayanan, Robust Speech Rate Estimation for Spontaneous Speech, 2007. https://sail.usc.edu/publications/files/WangTASLP2007%5B2%5D.pdf
92. Godsill, Rayner, Digital Audio Restoration: A Statistical Model-Based Approach, ch. 5 "Removal of Clicks", 1998. https://link.springer.com/chapter/10.1007/978-1-4471-1561-8_5
93. Muehlbauer, Automatic Audio Defect Detection (Bachelor thesis, JKU Linz), 2010. https://www.cp.jku.at/people/seyerlehner/supervised/muehlbauer_bakk.pdf
94. Alonso, Bogdanov, Essentia audio-problem detection algorithms (ClickDetector, DiscontinuityDetector, SaturationDetector, SNR, GapsDetector, StartStopCut, TruePeakDetector, HumDetector); Ozturk 2023 FSD50K audit, 2019. https://essentia.upf.edu/algorithms_reference.html
95. Eyben, Wöllmer, Schuller, openSMILE: the Munich versatile and fast open-source audio feature extractor (opensmile-python), 2010. https://github.com/audeering/opensmile-python
96. Tao, Shi, Shim, Watanabe, ANCHOR: Autoregressive Non-intrusive Chunk-Ordered Refinement for Joint Multi-Resolution Speech Quality Modeling, Interspeech 2026. https://arxiv.org/abs/2606.10233
97. Laroche, Dolson, Phase-vocoder: about this phasiness business, WASPAA 1997. https://www.semanticscholar.org/paper/Phase-vocoder:-about-this-phasiness-business-Laroche-Dolson/a0ec34451a98a298640ac0e548b1f6734ef6915a (DOI 10.1109/ASPAA.1997.625603)
98. Cooper, Huang, Tsao, Wang, Toda, Yamagishi, The VoiceMOS Challenge 2023: Zero-shot Subjective Speech Quality Prediction for Multiple Domains, 2023. https://arxiv.org/abs/2310.02640
99. Zhou, Zhang, Wang, Tian, Xu, Pyramid Feature Attention Network for Speech Resampling Detection, Applied Sciences 14(11):4803, 2024. https://doi.org/10.3390/app14114803 (listed by the survey as relevant to the PitchShift resampling cue; abstract not read by the verifier)
100. Reddy, Gopal, Cutler, DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors (P.808), 2021. https://arxiv.org/abs/2010.15258
