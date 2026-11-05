import json
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aadd.constants import CLASSES, DEGRADATION_CLASSES

REPO = Path(__file__).resolve().parent.parent
SRC = [REPO / "examples" / "speech" / "61-70968-0000_0.flac", REPO / "examples" / "degradations" / "orig.flac"]


def _have_v1() -> bool:
    """v1 weights come from the Hugging Face Hub (cached after the first download)."""
    try:
        from aadd.model import resolve_model_path

        return resolve_model_path("v1").exists()
    except Exception:  # noqa: BLE001 - offline
        return False


HAVE_V1 = _have_v1()

pytest.importorskip("audiomentations")


def test_degrade_signal_keeps_length_and_labels_align():
    import librosa

    from aadd.data.augment import build_transforms, degrade_signal

    y, sr = librosa.load(SRC[0], sr=16000)
    random.seed(3)
    np.random.seed(3)
    out, rows = degrade_signal(y, sr, build_transforms(), random.Random(3), p_aug=1.0)
    assert len(out) == (len(y) // sr) * sr
    assert len(rows) == len(y) // sr
    assert all(r["augmentation"] in DEGRADATION_CLASSES for r in rows)
    assert all(json.loads(r["parameters"]).get("should_apply") for r in rows)


def test_make_dataset_is_deterministic(tmp_path):
    from aadd.data.augment import make_dataset

    a = make_dataset([str(SRC[0])], tmp_path / "a", copies=2, seed=7, workers=1, progress=False)
    b = make_dataset([str(SRC[0])], tmp_path / "b", copies=2, seed=7, workers=1, progress=False)
    assert [r["counts"] for r in a] == [r["counts"] for r in b]
    xa, _ = sf.read(tmp_path / "a" / "audio" / "61-70968-0000_0_2.flac")
    xb, _ = sf.read(tmp_path / "b" / "audio" / "61-70968-0000_0_2.flac")
    assert np.array_equal(xa, xb)
    c = make_dataset([str(SRC[0])], tmp_path / "c", copies=2, seed=8, workers=1, progress=False)
    assert [r["counts"] for r in a] != [r["counts"] for r in c]
    assert (tmp_path / "a" / "manifest.csv").exists() and (tmp_path / "a" / "make_data_config.json").exists()
    assert a[0]["copy"] == 0 and a[0]["n_degraded"] == 0  # copy 0 is the clean original


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    from aadd.data.augment import make_dataset
    from aadd.data.dataset import featurize_dir
    from aadd.features import FeatureSpec

    d = tmp_path_factory.mktemp("ds")
    make_dataset([str(p) for p in SRC], d, copies=2, seed=0, workers=1, progress=False)
    featurize_dir(d, d / "mel.npz", FeatureSpec(kind="mel"), workers=1, progress=False)
    return d


def test_featurized_dataset(small_dataset):
    from aadd.data.dataset import ChunkArrayDataset, split_by_speaker

    ds = ChunkArrayDataset(small_dataset / "mel.npz")
    x, y = ds[0]
    assert tuple(x.shape) == (1, 128, 63) and y.shape == (8,) and y.sum() == 1
    assert ds.Y.sum(axis=1).min() == 1  # exactly one label per chunk (v1/v2 protocol)
    assert ds.classes == CLASSES
    tr, va = split_by_speaker(ds, val_fraction=0.5, seed=0)
    assert len(tr) + len(va) == len(ds) and not set(ds.speakers()[tr]) & set(ds.speakers()[va])


def test_train_small_cnn_and_eval_npz(small_dataset, tmp_path):
    from aadd.eval import evaluate_npz
    from aadd.train import train

    out = tmp_path / "m.pt"
    res = train(
        str(small_dataset / "mel.npz"),
        str(out),
        arch="small_cnn",
        epochs=2,
        batch_size=4,
        val_fraction=0.5,
        device="cpu",
        pretrained=False,
        log_path=str(tmp_path / "log.json"),
    )
    assert out.exists() and res["best_epoch"] >= 0 and len(res["history"]) == 2
    ev = evaluate_npz(str(out), str(small_dataset / "mel.npz"), device="cpu")
    assert ev["chunk_metrics"]["n"] > 0 and set(ev["chunk_metrics"]["per_class"]) == set(CLASSES)


@pytest.mark.skipif(not HAVE_V1, reason="v1 weights not available (offline?)")
def test_eval_dir_v1(small_dataset, tmp_path):
    from aadd.eval import evaluate_dir, write_report

    ev = evaluate_dir("v1", small_dataset, device="cpu", workers=0, progress=False)
    assert ev["n_files"] == 6 and ev["chunk_metrics"]["n"] > 0 and "spearman" in ev["file_level"]
    write_report(ev, tmp_path / "rep")
    assert (tmp_path / "rep.json").exists() and (tmp_path / "rep.md").read_text().startswith("# ")


def test_baselines_scores_shapes():
    from aadd.baselines import BASELINE_CLASSES, baseline_scores, baseline_scores_file, clipping_score

    x = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1
    s = baseline_scores(x, 16000)
    assert s.shape == (len(BASELINE_CLASSES),)
    assert clipping_score(np.clip(x, -0.05, 0.05)) > 0.1 > clipping_score(x)
    assert baseline_scores_file(str(SRC[0])).shape == (4, len(BASELINE_CLASSES))


def test_metrics():
    from aadd.metrics import compute_metrics, metrics_table

    y = np.eye(8)[[0, 1, 7, 7]]
    p = np.clip(y + 0.1, 0, 1)
    m = compute_metrics(y, p, CLASSES)
    assert m["f1_micro"] == 1.0 and m["single_label"] and m["argmax_accuracy"] == 1.0
    assert "| None |" in metrics_table(m)


@pytest.mark.skipif(not HAVE_V1, reason="v1 weights not available (offline?)")
def test_cli_scan(tmp_path, capsys):
    from aadd.cli import main

    rc = main(
        [
            "scan",
            str(REPO / "examples" / "degradations"),
            "--model",
            "v1",
            "--device",
            "cpu",
            "-q",
            "-o",
            str(tmp_path / "f.csv"),
            "--segments",
            str(tmp_path / "s.csv"),
            "--json",
            str(tmp_path / "r.json"),
        ]
    )
    assert rc == 0
    rows = (tmp_path / "f.csv").read_text().splitlines()
    assert len(rows) == 8 and rows[0].startswith("path,duration_s")
    j = json.loads((tmp_path / "r.json").read_text())
    assert j["classes"] == CLASSES and len(j["files"]) == 7


def test_cli_info(capsys):
    from aadd.cli import main

    assert main(["info"]) == 0
    assert "v1" in capsys.readouterr().out


def test_streamlit_app_runs_without_exceptions():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "aadd" / "app.py"), default_timeout=120).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value.startswith("AADD")
