import io
from pathlib import Path

import numpy as np
import pytest

from aadd.constants import CLASSES, NONE_CLASS
from aadd.features import FeatureSpec, load_audio, mel_db_v1, mel_features, png_features, render_mel_png, segment_audio

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "speech" / "61-70968-0000_4.flac"
CLIP_EXAMPLE = REPO / "examples" / "degradations" / "ClippingDistortion.flac"
V1 = REPO / "weights" / "aadd_v1_efficientnet_b0_png.pt"


def _pyplot_png(segment, sr):
    """The original v1 rendering code, verbatim (pyplot API, temp buffer)."""
    import librosa
    import librosa.display
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(4, 5))
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def test_segment_audio_drops_tail_by_default():
    y = np.zeros(16000 * 3 + 8000, dtype=np.float32)
    segs = segment_audio(y, 16000)
    assert [(s.start, s.end) for s in segs] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    segs = segment_audio(y, 16000, keep_tail=True)
    assert len(segs) == 4 and segs[-1].padded == 8000 and segs[-1].end == pytest.approx(3.5)


def test_segment_audio_short_file_is_padded():
    segs = segment_audio(np.ones(4000, dtype=np.float32), 16000)
    assert len(segs) == 1 and len(segs[0].samples) == 16000 and segs[0].padded == 12000


def test_v1_png_render_matches_original_code_pixel_for_pixel():
    y, sr = load_audio(EXAMPLE)
    seg = segment_audio(y, sr)[1].samples
    a = _pyplot_png(seg, sr)
    b = np.asarray(render_mel_png(mel_db_v1(seg, sr), sr))
    assert a.shape == b.shape == (385, 310, 3)
    assert np.array_equal(a, b)


def test_png_features_shape_and_range():
    y, sr = load_audio(EXAMPLE)
    x = png_features(segment_audio(y, sr)[0].samples, sr)
    assert x.shape == (3, 385, 310) and x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0


def test_mel_features_shape_and_range():
    y, sr = load_audio(EXAMPLE)
    x = mel_features(segment_audio(y, sr)[0].samples, sr)
    assert x.shape == (1, 128, 63) and x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() == pytest.approx(1.0)


def test_feature_spec_roundtrip():
    s = FeatureSpec(kind="mel", params={"hop_length": 160})
    assert FeatureSpec.from_dict(s.to_dict()) == s


@pytest.mark.skipif(not V1.exists(), reason="v1 weights not present")
def test_detector_v1_on_examples():
    from aadd.infer import Detector

    det = Detector("v1", device="cpu")
    assert det.classes == CLASSES and det.features.kind == "png"
    res = det.predict_file(CLIP_EXAMPLE)
    assert res.error is None and res.n_segments == 2 and res.probs.shape == (2, 8)
    assert "ClippingDistortion" in res.detected and res.max_probs["ClippingDistortion"] > 0.9
    row = res.file_row()
    assert row["status"] == "degraded" and row["aadd_count"] >= 2
    assert len(res.segment_rows()) == 2
    clean = det.predict_file(REPO / "examples" / "speech" / "61-70968-0000_0.flac")
    assert clean.is_clean and clean.probs[:, CLASSES.index(NONE_CLASS)].min() > 0.9


@pytest.mark.skipif(not V1.exists(), reason="v1 weights not present")
def test_detector_resamples_other_rates():
    from aadd.infer import Detector

    det = Detector("v1", device="cpu")
    y, sr = load_audio(EXAMPLE, sr=None)
    import librosa

    y44 = librosa.resample(y, orig_sr=sr, target_sr=44100)
    res = det.predict(y44, 44100)
    assert res.n_segments == 4 and res.sample_rate == 44100


@pytest.mark.skipif(not V1.exists(), reason="v1 weights not present")
def test_detector_error_row_for_bad_file(tmp_path):
    from aadd.infer import Detector

    bad = tmp_path / "x.flac"
    bad.write_bytes(b"not audio")
    res = Detector("v1", device="cpu").predict_file(bad)
    assert res.error and res.file_row()["status"] == "error" and res.n_segments == 0


def test_find_audio_files(tmp_path):
    from aadd.infer import find_audio_files

    (tmp_path / "a.flac").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.WAV").write_bytes(b"")
    (tmp_path / "sub" / "c.txt").write_bytes(b"")
    assert [Path(p).name for p in find_audio_files([tmp_path])] == ["a.flac", "b.WAV"]
    assert [Path(p).name for p in find_audio_files([tmp_path], recursive=False)] == ["a.flac"]
