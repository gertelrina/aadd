"""Render the README results tables from runs/eval_*.json and runs/speed_benchmark.json."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
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


def load(name):
    p = RUNS / name
    return json.load(open(p)) if p.exists() else None


def fmt(x, nd=3):
    return "—" if x is None or math.isnan(x) else f"{x:.{nd}f}"


def bold_best(values, nd=3):
    nums = [v for v in values if v is not None and not math.isnan(v)]
    best = max(nums) if nums else None
    return [f"**{fmt(v, nd)}**" if v is not None and v == best else fmt(v, nd) for v in values]


v1, v2, bl, sp = (
    load("eval_v1_test_clean.json"),
    load("eval_v2_test_clean.json"),
    load("eval_baselines_test_clean.json"),
    load("speed_benchmark.json"),
)
out = []

# 1. headline
out += [
    "### Models at a glance",
    "",
    "| model | input | F1 macro (8 classes) | F1 macro (7 degradations) | F1 micro | exact match per second | file-level Spearman | ms per audio second, CPU / MPS |",
    "|---|---|---|---|---|---|---|---|",
]
rows = {
    "`v1` (2025)": (
        "PNG 310×385",
        v1["chunk_metrics"]["f1_macro"],
        v1["chunk_metrics"]["f1_macro_degradations"],
        v1["chunk_metrics"]["f1_micro"],
        v1["chunk_metrics"]["exact_match"],
        v1["file_level"]["spearman"],
        f"{sp['v1/cpu']['ms_per_audio_second']:.0f} / {sp['v1/mps']['ms_per_audio_second']:.1f}",
    ),
    "`v2` (this release)": (
        "log-mel 1×128×63",
        v2["chunk_metrics"]["f1_macro"],
        v2["chunk_metrics"]["f1_macro_degradations"],
        v2["chunk_metrics"]["f1_micro"],
        v2["chunk_metrics"]["exact_match"],
        v2["file_level"]["spearman"],
        f"{sp['v2/cpu']['ms_per_audio_second']:.0f} / {sp['v2/mps']['ms_per_audio_second']:.1f}",
    ),
}
names = list(rows)
cols = list(zip(*[rows[n][1:6] for n in names]))
bolded = [bold_best(list(c)) for c in cols]
for i, name in enumerate(names):
    r = rows[name]
    out.append(f"| {name} | {r[0]} | " + " | ".join(b[i] for b in bolded) + f" | {r[6]} |")
out.append("")

# 2. per-class F1
out += [
    "### Per-class F1 (threshold 0.5)",
    "",
    "| class | support (seconds) | `v1` | `v2` | DSP baseline (best F1) |",
    "|---|---|---|---|---|",
]
for c in CLASSES:
    a, b = v1["chunk_metrics"]["per_class"][c], v2["chunk_metrics"]["per_class"][c]
    d = bl["per_class"].get(c, {}) if bl else {}
    vals = bold_best([a["f1"], b["f1"], d.get("best_f1")])
    out.append(f"| {c} | {a['support']:,} | " + " | ".join(vals) + " |")
out.append("")

# 3. per-class AUC + precision/recall of v2
out += [
    "### Per-class detail",
    "",
    "| class | `v1` ROC-AUC | `v2` ROC-AUC | DSP ROC-AUC | `v2` precision | `v2` recall | `v2` AP |",
    "|---|---|---|---|---|---|---|",
]
for c in CLASSES:
    a, b = v1["chunk_metrics"]["per_class"][c], v2["chunk_metrics"]["per_class"][c]
    d = bl["per_class"].get(c, {}) if bl else {}
    auc = bold_best([a["roc_auc"], b["roc_auc"], d.get("roc_auc")])
    out.append(f"| {c} | " + " | ".join(auc) + f" | {fmt(b['precision'])} | {fmt(b['recall'])} | {fmt(b['ap'])} |")
out.append("")

# 4. file level
out += ["### File level (15 720 files)", "", "| | `v1` | `v2` |", "|---|---|---|"]
for k, label in [
    ("spearman", "Spearman(true number of degraded seconds, predicted `aadd_count`)"),
    ("pearson", "Pearson"),
    ("mae", "mean absolute error of the count, seconds"),
]:
    out.append(f"| {label} | {fmt(v1['file_level'][k])} | {fmt(v2['file_level'][k])} |")
out.append("")

# 5. confusion of v2 (argmax)
cm = v2["chunk_metrics"]["confusion_argmax"]
short = ["Noise", "Pitch", "Stretch", "Shift", "Clip", "LPF", "HPF", "None"]
out += [
    "### `v2` confusion matrix (argmax over the 8 outputs; rows = true, columns = predicted)",
    "",
    "| true \\ pred | " + " | ".join(short) + " |",
    "|---|" + "---|" * len(short),
]
for i, c in enumerate(CLASSES):
    out.append(
        f"| {short[i]} | " + " | ".join(f"**{v:,}**" if i == j else f"{v:,}" for j, v in enumerate(cm[i])) + " |"
    )
out.append("")

# 6. speed
out += [
    "### Speed (300 test files, 2 106 s of audio, one process, Apple M3 Pro)",
    "",
    "| | `v1` | `v2` | speed-up |",
    "|---|---|---|---|",
]
for dev, label in [("cpu", "CPU"), ("mps", "Apple MPS")]:
    a, b = sp[f"v1/{dev}"], sp[f"v2/{dev}"]
    out.append(
        f"| {label}: ms per second of audio (× real time) | {a['ms_per_audio_second']:.1f} ({a['x_realtime']:.0f}×) | {b['ms_per_audio_second']:.1f} ({b['x_realtime']:.0f}×) | {a['ms_per_audio_second'] / b['ms_per_audio_second']:.1f}× |"
    )
text = "\n".join(out)
blocks = text.split("### ")[1:]
rendered = []
for i, b in enumerate(blocks):
    title, _, body = b.partition("\n")
    if i == 0:  # headline table stays visible
        rendered.append(f"### {title}\n{body.rstrip()}\n")
    else:
        rendered.append(f"<details><summary><b>{title}</b></summary>\n{body.rstrip()}\n\n</details>\n")
print("\n".join(rendered))
