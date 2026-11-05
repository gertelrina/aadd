"""Self-contained HTML dashboard for a scanned dataset.

    aadd scan DATA_DIR --json scan.json
    aadd report scan.json -o report.html

No external assets: inline SVG, CSS and ~40 lines of JS for tooltips. Light and
dark themes follow the OS setting. Folders directly above the audio files are
used as subsets ("subset" = parent directory name).
"""

from __future__ import annotations

import datetime as _dt
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aadd.constants import NONE_CLASS

PRETTY = {
    "noise": "Background noise",
    "music": "Music",
    "reverb": "Reverberation",
    "clipping": "Clipping",
    "lowpass": "Low-pass / narrow band",
    "highpass": "High-pass",
    "codec": "Codec artefacts",
    "packet_loss": "Packet loss",
    "hum": "Mains hum",
    "clicks": "Clicks",
    "pitch_shift": "Pitch shift",
    "time_stretch": "Time stretch",
    "AddGaussianNoise": "Gaussian noise",
    "PitchShift": "Pitch shift",
    "TimeStretch": "Time stretch",
    "Shift": "Discontinuity (shift)",
    "ClippingDistortion": "Clipping",
    "LowPassFilter": "Low-pass",
    "HighPassFilter": "High-pass",
}

CSS = """
.viz-root{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#e4e3df;--grid:#ecebe7;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#77766f;--series-1:#2a78d6;
--seq-0:#f0efec;--seq-1:#cde2fb;--seq-2:#9ec5f4;--seq-3:#6da7ec;--seq-4:#3987e5;--seq-5:#256abf;--seq-6:#184f95;--seq-7:#0d366b;
--good:#0ca30c;--warning:#fab219;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-0:#111110;
--surface-1:#1a1a19;--border:#2c2c2a;--grid:#262624;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#94938b;
--series-1:#3987e5;--seq-0:#262624;--seq-1:#104281;--seq-2:#184f95;--seq-3:#1c5cab;--seq-4:#256abf;--seq-5:#3987e5;--seq-6:#6da7ec;--seq-7:#b7d3f6}}
*{box-sizing:border-box}body{margin:0;background:var(--surface-0)}
.viz-root{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;color:var(--text-primary);
background:var(--surface-0);padding:28px 32px 48px;max-width:1280px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 2px}.sub{color:var(--text-secondary);font-size:13px}
.muted{color:var(--text-muted)}.grid{display:grid;gap:16px}.g6{grid-template-columns:repeat(6,1fr)}.g2{grid-template-columns:1fr 1fr}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.tile .v{font-size:26px;font-weight:650;letter-spacing:-.01em}.tile .l{color:var(--text-secondary);font-size:12.5px}
.tile .d{color:var(--text-muted);font-size:12px;margin-top:2px}
section{margin-top:16px}.head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
svg text{fill:var(--text-secondary);font-size:12px}svg .lbl{fill:var(--text-primary)}svg .val{fill:var(--text-secondary);font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--text-muted);font-weight:500;
border-bottom:1px solid var(--border);padding:6px 8px}td{border-bottom:1px solid var(--grid);padding:6px 8px;vertical-align:middle}
td.num{text-align:right;font-variant-numeric:tabular-nums}.chip{display:inline-block;border:1px solid var(--border);border-radius:999px;
padding:0 8px;margin:1px 3px 1px 0;font-size:12px;color:var(--text-secondary)}
.status{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:var(--text-secondary)}
.dot{width:8px;height:8px;border-radius:50%}.strip{display:flex;gap:1px;height:14px;min-width:160px}.strip i{flex:1;border-radius:2px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:6px 9px;font-size:12px;color:var(--text-primary);box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:9;max-width:320px}
.hit{cursor:default}.hit:hover{opacity:.85}details summary{cursor:pointer;color:var(--text-muted);font-size:12.5px;margin-top:8px}
@media (max-width:900px){.g6{grid-template-columns:repeat(3,1fr)}.g2{grid-template-columns:1fr}}
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');
if(!t){tip.style.display='none';return}tip.innerHTML=t.dataset.tip;tip.style.display='block';
const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8),y=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8);
tip.style.left=x+'px';tip.style.top=y+'px'});
"""


def _seq(v: float) -> str:
    """Sequential blue step for v in [0, 1]."""
    k = 0 if v <= 0.001 else min(7, 1 + int(v * 7))
    return f"var(--seq-{k})"


def _fmt_dur(sec: float) -> str:
    h = sec / 3600
    return f"{h:.1f} h" if h >= 1 else f"{sec / 60:.0f} min"


def _pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def _hbar(rows: list[tuple[str, float, str]], width: int = 560, row_h: int = 26, label_w: int = 170, fmt=_pct) -> str:
    """Horizontal bar chart: rows = (label, value in [0,1], tooltip html)."""
    vmax = max([v for _, v, _ in rows] + [1e-9])
    plot_w = width - label_w - 56
    h = row_h * len(rows) + 8
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    for i, (lab, v, tipx) in enumerate(rows):
        y = 4 + i * row_h
        w = max(2.0, plot_w * v / vmax) if v > 0 else 0
        out.append(
            f'<text class="lbl" x="{label_w - 10}" y="{y + row_h / 2 + 4}" text-anchor="end">{html.escape(lab)}</text>'
        )
        out.append(
            f'<rect x="{label_w}" y="{y + 5}" width="{plot_w}" height="{row_h - 10}" fill="var(--grid)" rx="4"/>'
        )
        if w:
            out.append(
                f'<rect class="hit" data-tip="{html.escape(tipx)}" x="{label_w}" y="{y + 5}" width="{w:.1f}" height="{row_h - 10}" fill="var(--series-1)" rx="4"/>'
            )
        out.append(f'<text class="val" x="{label_w + w + 6}" y="{y + row_h / 2 + 4}">{fmt(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def _histogram(values: np.ndarray, bins: int = 10, width: int = 560, height: int = 220) -> str:
    edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(np.clip(values, 0, 1), bins=edges)
    cmax = max(counts.max(), 1)
    left, bottom, top = 40, 34, 10
    pw, ph = width - left - 10, height - bottom - top
    bw = pw / bins
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for t in np.linspace(0, cmax, 4):
        y = top + ph - ph * t / cmax
        out.append(f'<line x1="{left}" x2="{width - 10}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{round(t)}</text>')
    for i, c in enumerate(counts):
        hgt = ph * c / cmax
        x = left + i * bw + 1
        lo, hi = int(edges[i] * 100), int(edges[i + 1] * 100)
        tipx = f"<b>{lo}–{hi}%</b> of the file degraded<br>{c} files"
        if c:
            out.append(
                f'<rect class="hit" data-tip="{tipx}" x="{x:.1f}" y="{top + ph - hgt:.1f}" width="{bw - 2:.1f}" height="{hgt:.1f}" fill="var(--series-1)" rx="3"/>'
            )
        out.append(f'<text x="{x + bw / 2 - 1:.1f}" y="{height - bottom + 16}" text-anchor="middle">{lo}</text>')
    out.append(
        f'<text x="{left + pw / 2}" y="{height - 4}" text-anchor="middle">share of the file’s seconds with any degradation, %</text>'
    )
    out.append("</svg>")
    return "".join(out)


def _heatmap(classes: list[str], M: np.ndarray, N: np.ndarray, width: int = 560) -> str:
    """M[i,j] = share of seconds with class i where class j is also detected (row-normalised)."""
    n = len(classes)
    if n < 2:
        return '<p class="muted">Not enough detections to show co-occurrence (needs two classes with enough seconds).</p>'
    lab_w = 160
    cell = (width - lab_w - 70) / n
    h = lab_w + n * cell
    out = [f'<svg viewBox="0 0 {width} {h + 4:.0f}" width="100%" role="img">']
    for j, c in enumerate(classes):
        x = lab_w + j * cell + cell / 2
        out.append(
            f'<text x="{x:.1f}" y="{lab_w - 8}" transform="rotate(-50 {x:.1f} {lab_w - 8})">{html.escape(PRETTY.get(c, c))}</text>'
        )
    for i, ci in enumerate(classes):
        y = lab_w + i * cell
        out.append(
            f'<text class="lbl" x="{lab_w - 8}" y="{y + cell / 2 + 4:.1f}" text-anchor="end">{html.escape(PRETTY.get(ci, ci))}</text>'
        )
        for j, cj in enumerate(classes):
            v = M[i, j]
            if i == j:
                out.append(
                    f'<rect x="{lab_w + j * cell + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="3" fill="none" stroke="var(--border)"/>'
                )
                continue
            tipx = f"<b>{html.escape(PRETTY.get(ci, ci))}</b> seconds: {int(N[i])}<br>also <b>{html.escape(PRETTY.get(cj, cj))}</b>: {_pct(v, 1)}"
            out.append(
                f'<rect class="hit" data-tip="{tipx}" x="{lab_w + j * cell + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="3" fill="{_seq(v) if N[i] else "var(--seq-0)"}"/>'
            )
    out.append("</svg>")
    return "".join(out)


def _strip(p: np.ndarray, labels: list[list[str]], start: list[float]) -> str:
    """Per-second strip coloured by max degradation probability."""
    n = len(p)
    step = max(1, int(np.ceil(n / 60)))
    cells = []
    for k in range(0, n, step):
        v = float(p[k : k + step].max())
        labs = sorted({c for L in labels[k : k + step] for c in L})
        tipx = f"{start[k]:.0f}–{start[min(k + step, n) - 1] + 1:.0f} s: " + (
            ", ".join(PRETTY.get(c, c) for c in labs) or "clean"
        )
        cells.append(f'<i data-tip="{html.escape(tipx)}" style="background:{_seq(v)}"></i>')
    return f'<div class="strip">{"".join(cells)}</div>'


def build_report(scan: dict, title: str | None = None, top_files: int = 25) -> str:
    classes = [c for c in scan["classes"] if c != NONE_CLASS]
    thr = float(scan.get("threshold", 0.5))
    files = scan["files"]
    C = len(classes)
    per_sub = defaultdict(
        lambda: {"files": 0, "sec": 0.0, "deg_sec": 0, "n_seg": 0, "deg_files": 0, "cls_sec": np.zeros(C)}
    )
    cls_sec = np.zeros(C)
    cls_files = np.zeros(C)
    co = np.zeros((C, C))
    total_dur = total_seg = deg_seg = deg_files = errors = 0
    file_frac, rows = [], []
    for f in files:
        fr = f["file"]
        sub = Path(f["path"]).parent.name or "."
        S = per_sub[sub]
        S["files"] += 1
        if fr.get("status") == "error":
            errors += 1
            continue
        segs = f["segments"]
        P = np.array([[s.get(f"p_{c}", 0.0) for c in classes] for s in segs]) if segs else np.zeros((0, C))
        D = P >= thr
        dur = float(f.get("duration_s", fr.get("duration_s", 0.0)))
        total_dur += dur
        S["sec"] += dur
        total_seg += len(P)
        S["n_seg"] += len(P)
        anyd = D.any(1)
        deg_seg += int(anyd.sum())
        S["deg_sec"] += int(anyd.sum())
        cls_sec += D.sum(0)
        S["cls_sec"] += D.sum(0)
        cls_files += D.any(0)
        co += D.T.astype(float) @ D.astype(float)
        if anyd.any():
            deg_files += 1
            S["deg_files"] += 1
        frac = float(anyd.mean()) if len(P) else 0.0
        file_frac.append(frac)
        labs = [[c for c, on in zip(classes, d) if on] for d in D]
        rows.append(
            (
                frac,
                f["path"],
                sub,
                dur,
                P.max(1) if len(P) else np.zeros(0),
                labs,
                [s["start_s"] for s in segs],
                D.any(0),
            )
        )
    n_ok = len(files) - errors
    diag = np.maximum(np.diag(co), 1e-9)
    M = co / diag[:, None]
    # co-occurrence only for classes seen in at least 20 s (or 0.5 % of the audio), ordered by prevalence
    min_sec = max(20, 0.005 * max(total_seg, 1))
    keep = [int(j) for j in np.argsort(-np.diag(co)) if co[j, j] >= min_sec]

    # ---- tiles
    tiles = [
        (f"{len(files):,}", "files analysed", f"{len(per_sub)} folders"),
        (_fmt_dur(total_dur), "audio analysed", f"{total_seg:,} one-second segments"),
        (_pct(deg_seg / max(total_seg, 1)), "of audio time degraded", f"{deg_seg:,} seconds"),
        (_pct(deg_files / max(n_ok, 1)), "of files have a degradation", f"{deg_files:,} files"),
        (f"{n_ok - deg_files:,}", "clean files", f"threshold p ≥ {thr:g}"),
        (f"{errors:,}", "unreadable files", "decode errors" if errors else "none"),
    ]
    tiles_html = "".join(
        f'<div class="card tile"><div class="v">{v}</div><div class="l">{l}</div><div class="d">{d}</div></div>'
        for v, l, d in tiles
    )

    order = np.argsort(-cls_sec)
    bar_time = [
        (
            PRETTY.get(classes[j], classes[j]),
            cls_sec[j] / max(total_seg, 1),
            f"<b>{PRETTY.get(classes[j], classes[j])}</b><br>{int(cls_sec[j]):,} s of {total_seg:,} ({_pct(cls_sec[j] / max(total_seg, 1), 1)})<br>in {int(cls_files[j]):,} files",
        )
        for j in order
    ]
    bar_files = [
        (
            PRETTY.get(classes[j], classes[j]),
            cls_files[j] / max(n_ok, 1),
            f"<b>{PRETTY.get(classes[j], classes[j])}</b><br>{int(cls_files[j]):,} of {n_ok:,} files ({_pct(cls_files[j] / max(n_ok, 1), 1)})",
        )
        for j in order
    ]
    table_cls = "".join(
        f"<tr><td>{html.escape(PRETTY.get(classes[j], classes[j]))}</td><td class='num'>{int(cls_sec[j]):,}</td><td class='num'>{_pct(cls_sec[j] / max(total_seg, 1), 1)}</td><td class='num'>{int(cls_files[j]):,}</td><td class='num'>{_pct(cls_files[j] / max(n_ok, 1), 1)}</td></tr>"
        for j in order
    )

    # ---- subsets
    sub_rows = []
    for name, S in sorted(per_sub.items(), key=lambda kv: -kv[1]["deg_sec"] / max(kv[1]["n_seg"], 1)):
        share = S["deg_sec"] / max(S["n_seg"], 1)
        top = np.argsort(-S["cls_sec"])[:3]
        chips = "".join(
            f'<span class="chip">{html.escape(PRETTY.get(classes[j], classes[j]))} {_pct(S["cls_sec"][j] / max(S["n_seg"], 1))}</span>'
            for j in top
            if S["cls_sec"][j] > 0
        )
        bar = f'<svg viewBox="0 0 120 12" width="120" height="12"><rect width="120" height="12" rx="4" fill="var(--grid)"/><rect width="{max(2, 120 * share) if share else 0:.1f}" height="12" rx="4" fill="var(--series-1)"/></svg>'
        sub_rows.append(
            f"<tr><td>{html.escape(name)}</td><td class='num'>{S['files']:,}</td><td class='num'>{_fmt_dur(S['sec'])}</td><td class='num'>{_pct(S['deg_files'] / max(S['files'], 1))}</td><td>{bar} <span class='muted'>{_pct(share)}</span></td><td>{chips or '<span class=muted>—</span>'}</td></tr>"
        )

    # ---- worst files
    rows.sort(key=lambda r: (-r[0], -int(r[7].sum()), r[1]))
    worst = []
    for frac, path, sub, dur, pmax, labs, starts, anycls in rows[:top_files]:
        name = Path(path).name
        chips = "".join(
            f'<span class="chip">{html.escape(PRETTY.get(c, c))}</span>' for c, on in zip(classes, anycls) if on
        )
        worst.append(
            f"<tr><td title='{html.escape(path)}'>{html.escape(name)}</td><td class='muted'>{html.escape(sub)}</td><td class='num'>{dur:.1f} s</td><td class='num'>{_pct(frac)}</td><td>{_strip(pmax, labs, starts) if len(pmax) else ''}</td><td>{chips}</td></tr>"
        )

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    title = title or "Audio degradation report"
    model = html.escape(str(scan.get("model", "")))
    body = f"""
<div class="viz-root">
<h1>{html.escape(title)}</h1>
<div class="sub">model <b>{html.escape(Path(model).name or model)}</b> · threshold {thr:g} · generated {now} by <code>aadd report</code></div>
<section class="grid g6">{tiles_html}</section>
<section class="grid g2">
 <div class="card"><div class="head"><h2>Share of audio time affected</h2><span class="muted">% of all seconds</span></div>{_hbar(bar_time)}
  <details><summary>Show as table</summary><table><tr><th>degradation</th><th class=num>seconds</th><th class=num>% time</th><th class=num>files</th><th class=num>% files</th></tr>{table_cls}</table></details></div>
 <div class="card"><div class="head"><h2>Files affected</h2><span class="muted">% of files with ≥ 1 detected second</span></div>{_hbar(bar_files)}</div>
</section>
<section class="grid g2">
 <div class="card"><div class="head"><h2>How degraded is each file?</h2><span class="muted">{n_ok:,} files</span></div>{_histogram(np.array(file_frac))}</div>
 <div class="card"><div class="head"><h2>Degradations that occur together</h2><span class="muted">row: share of its seconds where the column class is also present · classes with ≥ {int(min_sec)} s</span></div>{_heatmap([classes[j] for j in keep], M[np.ix_(keep, keep)], np.diag(co)[keep])}</div>
</section>
<section class="card"><div class="head"><h2>By folder</h2><span class="muted">sorted by share of degraded time</span></div>
<table><tr><th>folder</th><th class=num>files</th><th class=num>audio</th><th class=num>files with issues</th><th>degraded time</th><th>top degradations (% time)</th></tr>{"".join(sub_rows)}</table></section>
<section class="card"><div class="head"><h2>Most degraded files</h2><span class="muted">top {min(top_files, len(rows))} of {len(rows):,} · strip = max probability per second, darker = more certain</span></div>
<table><tr><th>file</th><th>folder</th><th class=num>length</th><th class=num>degraded</th><th>timeline</th><th>detected</th></tr>{"".join(worst)}</table></section>
</div><div id="tip"></div>"""
    return f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}<script>{JS}</script></body></html>"


def write_report(scan_json: str, out: str, title: str | None = None, top_files: int = 25) -> str:
    scan = json.loads(Path(scan_json).read_text())
    Path(out).write_text(build_report(scan, title=title, top_files=top_files))
    return out
