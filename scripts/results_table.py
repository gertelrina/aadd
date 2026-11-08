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


def bold_best(values, nd=3, lower=False):
    nums = [v for v in values if v is not None and not math.isnan(v)]
    best = (min(nums) if lower else max(nums)) if nums else None
    return [f"**{fmt(v, nd)}** 🏆" if v is not None and v == best else fmt(v, nd) for v in values]


v1, v2, bl, sp = (
    load("eval_v1_test_clean.json"),
    load("eval_v2_test_clean.json"),
    load("eval_baselines_test_clean.json"),
    load("speed_benchmark.json"),
)
nq = load("nisqa_test_clean.json") if (RUNS / "nisqa_test_clean.json").exists() else None
out = []

# 1. headline
out += [
    "### v1 and v2 at a glance" + (" (NISQA for reference)" if nq else ""),
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
    "`v2`": (
        "log-mel 1×128×63",
        v2["chunk_metrics"]["f1_macro"],
        v2["chunk_metrics"]["f1_macro_degradations"],
        v2["chunk_metrics"]["f1_micro"],
        v2["chunk_metrics"]["exact_match"],
        v2["file_level"]["spearman"],
        f"{sp['v2/cpu']['ms_per_audio_second']:.0f} / {sp['v2/mps']['ms_per_audio_second']:.1f}",
    ),
}
if nq:
    nc = sp.get("nisqa/cpu", {}).get("ms_per_audio_second")
    rows["NISQA (quality score, no classes)"] = (
        "mel spectrogram",
        None,
        None,
        None,
        None,
        nq["file_level"]["spearman_n_degraded_vs_5_minus_mos"],
        f"{nc:.1f} / —" if nc else "—",
    )
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
out += [
    "### File level (15 720 files)",
    "",
    "| | `v1` | `v2` |" + (" NISQA |" if nq else ""),
    "|---|---|---|" + ("---|" if nq else ""),
]
nqk = {"spearman": "spearman_n_degraded_vs_5_minus_mos", "pearson": "pearson_n_degraded_vs_5_minus_mos"}
for k, label in [
    ("spearman", "Spearman(true number of degraded seconds, predicted `aadd_count`; for NISQA 5 − MOS)"),
    ("pearson", "Pearson"),
    ("mae", "mean absolute error of the count, seconds"),
]:
    vals = [v1["file_level"][k], v2["file_level"][k]] + ([nq["file_level"].get(nqk.get(k, ""))] if nq else [])
    out.append(f"| {label} | " + " | ".join(bold_best(vals, lower=k == "mae")) + " |")
if nq:
    f = nq["file_level"]
    out += [
        "",
        (
            f"NISQA MOS: {f['mean_mos_clean_copy']:.2f} on the clean copies, {f['mean_mos_degraded']:.2f} on files with "
            f'degraded seconds; ROC-AUC for "file has any degraded second" {f["auc_any_degraded_vs_clean_copy"]:.3f}. '
            "A quality score falls when something is wrong, but it does not count or name the degradations."
        ),
    ]
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
setup = sp.get("_setup", {})
models = [m for m in ("v1", "v2", "v3.1", "nisqa") if f"{m}/cpu" in sp]
out += [
    "### Speed: "
    + ", ".join(m if m != "nisqa" else "NISQA" for m in models[:-1])
    + f" and {models[-1].replace('nisqa', 'NISQA')} on the same 300 files",
    "",
    "| | " + " | ".join("NISQA" if m == "nisqa" else f"`{m}`" for m in models) + " |",
    "|---|" + "---|" * len(models),
]
for dev, label in [("cpu", "CPU"), ("mps", "Apple GPU (MPS)")]:
    cells = [
        f"{sp[f'{m}/{dev}']['ms_per_audio_second']:.1f} ms ({sp[f'{m}/{dev}']['x_realtime']:.0f}× real time)"
        if f"{m}/{dev}" in sp
        else "—"
        for m in models
    ]
    out.append(f"| {label}, per second of audio | " + " | ".join(cells) + " |")
for dev, label in [("cpu", "CPU"), ("mps", "Apple GPU (MPS)")]:
    fastest = min(sp[f"{m}/{dev}"]["ms_per_audio_second"] for m in models if f"{m}/{dev}" in sp)
    cells = [
        "—"
        if f"{m}/{dev}" not in sp
        else f"**{sp[f'{m}/{dev}']['ms_per_audio_second'] * 3.6:.0f} s** 🏆"
        if sp[f"{m}/{dev}"]["ms_per_audio_second"] == fastest
        else f"{sp[f'{m}/{dev}']['ms_per_audio_second'] * 3.6:.0f} s"
        for m in models
    ]
    out.append(f"| {label}, one hour of audio | " + " | ".join(cells) + " |")
if setup:
    out += [
        "",
        (
            f"Measured on {setup['machine']}, {setup['os']}, Python {setup['python']}, PyTorch {setup['torch']} "
            f"({setup['torch_cpu_threads']} CPU threads). Data: {setup['data']}. Method: {setup['method']}. "
            "v3.1 processes 8-s windows in one forward pass, which is why it is faster than v2 on CPU despite being 20× larger."
            + (
                " NISQA: the torchmetrics implementation, CPU only; it outputs five quality scores per file, not classes or a timeline."
                if "nisqa/cpu" in sp
                else ""
            )
        ),
    ]
text = "\n".join(out)
blocks = text.split("### ")[1:]
headline = "### " + blocks[0].rstrip() + "\n"
if "--details" not in sys.argv:
    print(headline)  # the table shown in the main README
else:
    # everything else goes to runs/README.md, next to the JSON reports it is built from
    notes = (
        (RUNS / "notes" / "how_to_read_v1_v2.md").read_text().strip()
        if (RUNS / "notes" / "how_to_read_v1_v2.md").exists()
        else ""
    )
    doc = [
        "# Evaluation details",
        "",
        "Generated by `python scripts/results_table.py runs --details > runs/README.md` from the JSON reports in this folder.",
        "Summary tables are in the main [README](../README.md#results).",
        "",
        "v1 and v2 are evaluated on the regenerated LibriSpeech test-clean set: 2 620 source files × (1 clean + 5 degraded",
        "copies) = 15 720 files, 108 936 one-second chunks, 41.6 % degraded, one label per second, threshold 0.5.",
        "",
    ]
    for b in blocks[1:]:
        title, _, body = b.partition("\n")
        doc += [f"## {title}", body.rstrip(), ""]
    if notes:
        doc += ["## How to read the v1 / v2 numbers", "", notes, ""]
    doc += [
        "## Files",
        "",
        "| file | content |",
        "|---|---|",
        "| `eval_v1_test_clean.json`, `eval_v2_test_clean.json` | v1 / v2 on the synthetic test set (per-class P / R / F1 / AUC / AP, confusion, file level) |",
        "| `eval_baselines_test_clean.json` | DSP baselines on the same set |",
        "| `eval_v3.json`, `eval_v3_1.json` | v3.0 / v3.1 on held-out generated sets and real out-of-domain sets |",
        "| `cross_eval.json` | v3.1 on the v1/v2 test set; v1 / v2 on real recordings |",
        "| `speed_benchmark.json` | timing of all models with the hardware and method |",
        "| `nisqa_ood.json`, `nisqa_test_clean.json` | NISQA as a reference on the same real out-of-domain sets and on the v1/v2 test set |",
        "| `fsd50k_noise_pool.json.gz` | FSD50K clip ids, split and licence of the v3.1 noise pool (CC0 / CC BY only, no music or voice labels) |",
        "| `v2_train_log.json`, `v3_train_log.json` | training curves |",
    ]
    print("\n".join(doc))
