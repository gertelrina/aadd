"""Command-line interface: ``aadd <command> ...``.

Commands
--------
scan                score audio files / folders -> CSV (per file + per second) or JSON
app                 launch the Streamlit UI
info                list bundled models and the compute device
make-data           build the synthetic degradation dataset from clean audio (v1/v2 protocol)
featurize           precompute chunk features (mel / png) for a make-data directory
train               train a model on featurized data
eval                evaluate a model (or the DSP baselines) on a make-data directory / .npz
convert-checkpoint  convert a legacy v1 checkpoint (with optimizer state) to the slim format
export-onnx         export a model to ONNX
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from aadd import __version__
from aadd.constants import CLASSES


def _add_model_args(p: argparse.ArgumentParser, default: str = "v2"):
    p.add_argument("--model", default=default, help="registry name (v1, v2) or checkpoint path [%(default)s]")
    p.add_argument("--threshold", type=float, default=0.5, help="detection probability threshold [%(default)s]")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    p.add_argument("--batch-size", type=int, default=64)


def cmd_scan(a: argparse.Namespace) -> int:
    from aadd.infer import Detector, find_audio_files

    files = find_audio_files(a.paths, recursive=not a.no_recursive, extensions=a.ext.split(","))
    if not files:
        print("no audio files found", file=sys.stderr)
        return 1
    det = Detector(
        a.model,
        device=a.device,
        threshold=a.threshold,
        batch_size=a.batch_size,
        keep_tail=a.keep_tail,
        resolution=a.resolution,
    )
    print(f"model={a.model} features={det.features.kind} device={det.device} files={len(files)}", file=sys.stderr)
    file_rows, seg_rows, dicts = [], [], []
    for res in det.scan(files, workers=a.workers, progress=not a.quiet):
        file_rows.append(res.file_row())
        if a.segments:
            seg_rows.extend(res.segment_rows())
        if a.json:
            dicts.append(res.to_dict())
        if not a.quiet and not a.out:
            d = res.detected
            tag = (
                "ERROR " + res.error
                if res.error
                else ("clean" if not d else ", ".join(f"{k} {v:.2f}" for k, v in sorted(d.items())))
            )
            print(f"{res.path}\t{res.duration:.1f}s\t{res.degraded_segments}/{res.n_segments} degraded s\t{tag}")
    if a.out:
        _write_csv(a.out, file_rows)
        print(f"wrote {a.out} ({len(file_rows)} files)", file=sys.stderr)
    if a.segments:
        _write_csv(a.segments, seg_rows)
        print(f"wrote {a.segments} ({len(seg_rows)} segments)", file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": a.model, "threshold": a.threshold, "classes": det.classes, "files": dicts}, f, indent=1)
        print(f"wrote {a.json}", file=sys.stderr)
    n_deg = sum(1 for r in file_rows if r["status"] == "degraded")
    n_err = sum(1 for r in file_rows if r["status"] == "error")
    print(
        f"summary: {len(file_rows)} files, {n_deg} degraded, {len(file_rows) - n_deg - n_err} clean, {n_err} errors",
        file=sys.stderr,
    )
    return 0


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def cmd_app(a: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).resolve().parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port)]
    if a.headless:
        cmd += ["--server.headless", "true"]
    cmd += ["--"] + (["--model", a.model] if a.model else []) + (["--scan", *a.scan] if a.scan else [])
    return subprocess.call(cmd)


def cmd_info(a: argparse.Namespace) -> int:
    import torch

    from aadd.model import REGISTRY, WEIGHTS_DIR, pick_device

    print(f"aadd {__version__}  torch {torch.__version__}  device {pick_device(a.device)}")
    print(f"classes: {CLASSES}")
    print(f"weights dir: {WEIGHTS_DIR}")
    for name, spec in REGISTRY.items():
        p = WEIGHTS_DIR / spec["file"]
        status = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "MISSING"
        print(f"  {name}: {spec['file']} [{status}] - {spec['description']}")
        if p.exists() and a.verbose:
            from aadd.model import load_checkpoint

            _, ck = load_checkpoint(name)
            print(
                "     "
                + json.dumps(
                    {k: v for k, v in ck["meta"].items() if k in ("epoch", "trained", "trained_on", "config")},
                    default=str,
                )
            )
    return 0


def cmd_make_data(a: argparse.Namespace) -> int:
    from aadd.data.augment import list_librispeech, make_dataset
    from aadd.infer import find_audio_files

    src = list_librispeech(a.src) if a.librispeech else find_audio_files([a.src])
    if not src:
        print("no source audio found", file=sys.stderr)
        return 1
    print(
        f"{len(src)} source files -> {a.out} (copies={a.copies}, seed={a.seed}, p_aug={a.p_aug}, stretch={a.stretch_method})",
        file=sys.stderr,
    )
    res = make_dataset(
        src,
        a.out,
        copies=a.copies,
        seed=a.seed,
        p_aug=a.p_aug,
        stretch_method=a.stretch_method,
        workers=a.workers,
        limit=a.limit,
    )
    n = sum("skipped" not in r for r in res)
    chunks = sum(r.get("n_chunks", 0) for r in res)
    degr = sum(r.get("n_degraded", 0) for r in res)
    print(f"wrote {n} files, {chunks} chunks, {degr} degraded chunks ({degr / max(chunks, 1):.1%})", file=sys.stderr)
    return 0


def cmd_featurize(a: argparse.Namespace) -> int:
    from aadd.data.dataset import featurize_dir
    from aadd.features import FeatureSpec

    spec = FeatureSpec(kind=a.features)
    info = featurize_dir(a.data, a.out, spec, workers=a.workers, limit=a.limit)
    print(json.dumps(info))
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from aadd.train import train

    res = train(
        a.train,
        a.out,
        test_npz=a.test,
        arch=a.arch,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        val_fraction=a.val_fraction,
        seed=a.seed,
        device=a.device,
        workers=a.workers,
        pretrained=not a.no_pretrained,
        spec_mask=not a.no_spec_mask,
        name=a.name,
        log_path=a.log,
        limit=a.limit,
    )
    print(f"best epoch {res['best_epoch']} ({res['best']:.4f}) -> {res['out']}  [{res['seconds'] / 60:.1f} min]")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from aadd.eval import evaluate_baselines, evaluate_dir, evaluate_npz, write_report
    from aadd.metrics import metrics_table

    if a.baselines:
        res = evaluate_baselines(a.data, limit=a.limit, workers=a.workers or None)
        title = "DSP baselines"
    elif str(a.data).endswith(".npz"):
        res = evaluate_npz(a.model, a.data, threshold=a.threshold, device=a.device, batch_size=a.batch_size)
        title = f"{a.model} on {a.data}"
    else:
        res = evaluate_dir(
            a.model,
            a.data,
            threshold=a.threshold,
            workers=a.workers,
            limit=a.limit,
            device=a.device,
            batch_size=a.batch_size,
            save_predictions=a.save_predictions,
        )
        title = f"{a.model} on {a.data}"
    if a.out:
        write_report(res, a.out, title=title)
        print(f"wrote {Path(a.out).with_suffix('.json')} and .md")
    if "chunk_metrics" in res:
        print(metrics_table(res["chunk_metrics"], list(res["chunk_metrics"]["per_class"].keys())))
        print("file level:", json.dumps(res["file_level"]))
    else:
        for c, v in res["per_class"].items():
            print(f"{c:20s} AUC {v['roc_auc']:.3f}  AP {v['ap']:.3f}  bestF1 {v['best_f1']:.3f}  (n={v['support']})")
    return 0


def cmd_convert_checkpoint(a: argparse.Namespace) -> int:
    from aadd.features import FeatureSpec
    from aadd.model import load_checkpoint, save_checkpoint

    model, ck = load_checkpoint(a.src, "cpu")
    save_checkpoint(
        a.out,
        model,
        ck["arch"],
        ck["in_channels"],
        FeatureSpec.from_dict(ck["features"]),
        classes=ck["classes"],
        meta=ck.get("meta", {}),
    )
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    return 0


def cmd_export_onnx(a: argparse.Namespace) -> int:
    import torch

    from aadd.model import load_checkpoint

    model, ck = load_checkpoint(a.model, "cpu")
    kind = ck["features"]["kind"]
    shape = (1, 3, 385, 310) if kind == "png" else (1, 1, 128, 63)
    dummy = torch.zeros(*shape)
    torch.onnx.export(
        model,
        dummy,
        a.out,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=a.opset,
    )
    print(f"wrote {a.out} (input {kind} {shape[1:]}, apply sigmoid to logits; classes {ck['classes']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aadd", description="Automatic Audio Degradation Detection")
    p.add_argument("--version", action="version", version=f"aadd {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="score audio files / folders")
    s.add_argument("paths", nargs="+", help="audio files or directories")
    _add_model_args(s)
    s.add_argument("--out", "-o", help="per-file CSV")
    s.add_argument("--segments", help="per-second CSV")
    s.add_argument("--json", help="full JSON output")
    s.add_argument("--workers", type=int, default=0, help="feature-extraction processes (0 = in-process)")
    s.add_argument("--ext", default="flac,wav,mp3,ogg,opus,m4a", help="extensions to scan [%(default)s]")
    s.add_argument("--no-recursive", action="store_true")
    s.add_argument("--keep-tail", action="store_true", help="also score a trailing partial second (zero-padded)")
    s.add_argument(
        "--resolution", type=float, default=1.0, help="segment length in seconds for frame models (v3) [%(default)s]"
    )
    s.add_argument("--quiet", "-q", action="store_true")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("app", help="launch the Streamlit UI")
    s.add_argument("--port", type=int, default=8501)
    s.add_argument("--headless", action="store_true")
    s.add_argument("--model", default=None)
    s.add_argument("--scan", nargs="*", default=None, help="files/folders to analyse automatically on start")
    s.set_defaults(func=cmd_app)

    s = sub.add_parser("info", help="models and device")
    s.add_argument("--device", default=None)
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("make-data", help="build the synthetic degradation dataset")
    s.add_argument("src", help="LibriSpeech split dir (e.g. LibriSpeech/dev-clean) or any folder of clean audio")
    s.add_argument("--out", required=True)
    s.add_argument("--copies", type=int, default=5, help="degraded copies per source file [%(default)s]")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--p-aug", type=float, default=0.5, help="per-second probability of a degradation [%(default)s]")
    s.add_argument(
        "--stretch-method", default="librosa_phase_vocoder", choices=["librosa_phase_vocoder", "signalsmith_stretch"]
    )
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--limit", type=int, default=None, help="only the first N source files")
    s.add_argument(
        "--no-librispeech", dest="librispeech", action="store_false", help="src is a flat folder, not LibriSpeech"
    )
    s.set_defaults(func=cmd_make_data)

    s = sub.add_parser("featurize", help="precompute chunk features")
    s.add_argument("data", help="make-data directory")
    s.add_argument("--out", required=True, help="output .npz")
    s.add_argument("--features", default="mel", choices=["mel", "png"])
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_featurize)

    s = sub.add_parser("train", help="train a model")
    s.add_argument("--train", required=True, help="featurized .npz")
    s.add_argument("--test", default=None, help="featurized .npz reported after every epoch (not used for selection)")
    s.add_argument("--out", required=True, help="checkpoint path")
    s.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "small_cnn"])
    s.add_argument("--epochs", type=int, default=10)
    s.add_argument("--batch-size", type=int, default=128)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--weight-decay", type=float, default=1e-2)
    s.add_argument("--val-fraction", type=float, default=0.1, help="speaker-disjoint validation share of --train")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None)
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--no-pretrained", action="store_true")
    s.add_argument("--no-spec-mask", action="store_true")
    s.add_argument("--name", default="v2")
    s.add_argument("--log", default=None, help="JSON training log")
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="evaluate a model or the DSP baselines")
    s.add_argument("data", help="make-data directory or featurized .npz")
    _add_model_args(s)
    s.add_argument("--baselines", action="store_true", help="evaluate the DSP baselines instead of a model")
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--out", help="report path without extension (.json + .md written)")
    s.add_argument("--save-predictions", help="per-chunk CSV of probabilities")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("convert-checkpoint", help="legacy .pth -> slim checkpoint")
    s.add_argument("src")
    s.add_argument("out")
    s.set_defaults(func=cmd_convert_checkpoint)

    s = sub.add_parser("export-onnx", help="export to ONNX")
    s.add_argument("--model", default="v2")
    s.add_argument("--out", required=True)
    s.add_argument("--opset", type=int, default=17)
    s.set_defaults(func=cmd_export_onnx)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
