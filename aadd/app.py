"""Streamlit UI for AADD. Run with ``aadd app`` or ``streamlit run aadd/app.py``."""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from aadd import __version__
from aadd.constants import DEGRADATION_CLASSES, NONE_CLASS
from aadd.infer import Detector, FileResult, find_audio_files
from aadd.model import REGISTRY, WEIGHTS_DIR

REPO = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO / "examples"


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--scan", nargs="*", default=None, help="files/folders to analyse automatically on start")
    a, _ = p.parse_known_args(sys.argv[1:])
    return a


@st.cache_resource(show_spinner="Loading model…")
def get_detector(model: str, device: str | None) -> Detector:
    return Detector(model, device=device, threshold=0.5)


def available_models() -> list[str]:
    return [n for n, s in REGISTRY.items() if (WEIGHTS_DIR / s["file"]).exists()]


def timeline_figure(res: FileResult):
    import plotly.express as px

    classes = [c for c in res.classes if c != NONE_CLASS] + [NONE_CLASS]
    idx = [res.classes.index(c) for c in classes]
    z = res.probs[:, idx].T
    fig = px.imshow(
        z,
        x=[f"{s:.0f}–{e:.0f}" for s, e in zip(res.starts, res.ends)],
        y=classes,
        zmin=0,
        zmax=1,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels={"x": "second", "y": "", "color": "p"},
        text_auto=".2f",
    )
    fig.update_layout(height=260, margin={"l": 0, "r": 0, "t": 10, "b": 0}, coloraxis_showscale=False)
    return fig


def main() -> None:
    cli = _args()
    st.set_page_config(page_title="AADD — Audio Degradation Detection", layout="wide")
    st.title("AADD — Automatic Audio Degradation Detection")
    st.caption(
        "Per-second detection of Gaussian noise, pitch shift, time stretch, circular shift, clipping, "
        "low-pass and high-pass filtering in speech. Trained on degraded LibriSpeech; "
        f"aadd {__version__}."
    )

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("Settings")
        models = available_models()
        if not models:
            st.error(f"No weights found in {WEIGHTS_DIR}")
            st.stop()
        default = cli.model if cli.model in models else ("v2" if "v2" in models else models[0])
        model_name = st.selectbox(
            "Model",
            models,
            index=models.index(default),
            help="\n".join(f"{n}: {REGISTRY[n]['description']}" for n in models),
        )
        threshold = st.slider("Detection threshold", 0.05, 0.95, 0.5, 0.05)
        keep_tail = st.checkbox("Score trailing partial second", value=False)
        device = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)
        det = get_detector(model_name, None if device == "auto" else device)
        det.threshold = threshold
        det.keep_tail = keep_tail
        st.write(f"features: `{det.features.kind}`, device: `{det.device}`")
        st.markdown("---")
        st.markdown("**Classes**")
        st.markdown("\n".join(f"- {c}" for c in DEGRADATION_CLASSES))
        st.markdown(
            "`Shift` (circular roll inside a second) is practically undetectable from a spectrogram; expect low scores for it."
        )

    # ---------------- input ----------------
    tab_in, tab_res, tab_stats, tab_gallery = st.tabs(
        ["1 · Input", "2 · Results", "3 · Dataset statistics", "Degradation gallery"]
    )

    with tab_in:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Upload files")
            uploads = st.file_uploader(
                "Audio files", type=["flac", "wav", "mp3", "ogg", "opus", "m4a"], accept_multiple_files=True
            )
        with c2:
            st.subheader("…or scan a folder on this machine")
            folder = st.text_input("Folder path", value="", placeholder="/path/to/audio")
            recursive = st.checkbox("Recursive", value=True)
            max_files = st.number_input("Max files", min_value=1, max_value=100000, value=500)
        st.subheader("…or try the bundled examples")
        examples = find_audio_files([EXAMPLES_DIR]) if EXAMPLES_DIR.exists() else []
        chosen_examples = st.multiselect(
            "Examples", examples, format_func=lambda p: str(Path(p).relative_to(EXAMPLES_DIR))
        )
        run = st.button("▶ Analyze", type="primary")

    auto = bool(cli.scan) and "results" not in st.session_state
    if run or auto:
        paths: list[str] = []
        tmpdir = Path(tempfile.mkdtemp(prefix="aadd_"))
        for up in uploads or []:
            p = tmpdir / up.name
            p.write_bytes(up.getbuffer())
            paths.append(str(p))
        if folder.strip():
            try:
                found = find_audio_files([folder.strip()], recursive=recursive)
                paths.extend(found[: int(max_files)])
                if len(found) > max_files:
                    st.warning(f"{len(found)} files found, only the first {int(max_files)} are scanned")
            except FileNotFoundError:
                st.error(f"Folder not found: {folder}")
        paths.extend(chosen_examples)
        if auto:
            paths.extend(find_audio_files(cli.scan))
        if not paths:
            st.warning("Nothing to analyze.")
        else:
            results: list[FileResult] = []
            bar = st.progress(0.0, text="Analyzing…")
            for i, res in enumerate(det.scan(paths)):
                results.append(res)
                bar.progress((i + 1) / len(paths), text=f"{i + 1}/{len(paths)} {Path(res.path).name}")
            bar.empty()
            st.session_state["results"] = results
            st.session_state["threshold"] = threshold
            st.success(f"Done: {len(results)} files. See the Results tab.")

    results: list[FileResult] = st.session_state.get("results", [])
    for r in results:  # live threshold without re-running inference
        r.threshold = threshold

    # ---------------- results ----------------
    with tab_res:
        if not results:
            st.info("Run an analysis first.")
        else:
            df = pd.DataFrame([r.file_row() for r in results])
            df["name"] = df["path"].map(lambda p: Path(p).name)
            cols = [
                "name",
                "status",
                "duration_s",
                "n_segments",
                "degraded_segments",
                "degraded_fraction",
                "aadd_count",
                "detected",
            ] + [f"p_{c}" for c in DEGRADATION_CLASSES]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
            seg_df = pd.DataFrame([row for r in results for row in r.segment_rows()])
            d1, d2 = st.columns(2)
            d1.download_button("⬇ per-file CSV", df.to_csv(index=False).encode(), "aadd_files.csv", "text/csv")
            d2.download_button("⬇ per-second CSV", seg_df.to_csv(index=False).encode(), "aadd_segments.csv", "text/csv")
            st.markdown("### Per-file detail")
            show_mel = st.checkbox("Show the mel spectrogram", value=False)
            for r in results:
                label = "🟢" if r.is_clean else ("🔴" if r.error else "🟠")
                with st.expander(
                    f"{label} {Path(r.path).name} — {r.degraded_segments}/{r.n_segments} degraded s — {', '.join(sorted(r.detected)) or 'clean'}"
                ):
                    if r.error:
                        st.error(r.error)
                        continue
                    st.audio(r.path)
                    st.plotly_chart(timeline_figure(r), use_container_width=True, key=f"tl_{r.path}")
                    if show_mel:
                        st.image(_mel_image(r.path), use_container_width=True)
                    st.table(
                        pd.DataFrame(
                            {
                                "second": [f"{s:.0f}–{e:.0f}" for s, e in zip(r.starts, r.ends)],
                                "labels": [", ".join(l) or "—" for l in r.segment_labels()],
                                "p(None)": np.round(r.probs[:, r.classes.index(NONE_CLASS)], 3),
                            }
                        )
                    )

    # ---------------- stats ----------------
    with tab_stats:
        if not results:
            st.info("Run an analysis first.")
        else:
            import plotly.express as px

            ok = [r for r in results if not r.error]
            total_seg = sum(r.n_segments for r in ok)
            dirty_seg = sum(r.degraded_segments for r in ok)
            dirty_files = sum(1 for r in ok if not r.is_clean)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Files", len(ok))
            m2.metric("Degraded files", f"{dirty_files} ({dirty_files / max(len(ok), 1):.0%})")
            m3.metric("Seconds analysed", total_seg)
            m4.metric("Degraded seconds", f"{dirty_seg} ({dirty_seg / max(total_seg, 1):.0%})")
            counts = {c: 0 for c in DEGRADATION_CLASSES}
            for r in ok:
                for labs in r.segment_labels():
                    for c in labs:
                        counts[c] += 1
            cdf = pd.DataFrame({"degradation": list(counts), "seconds": list(counts.values())})
            st.plotly_chart(
                px.bar(cdf, x="degradation", y="seconds", title="Detected degradations (seconds)"),
                use_container_width=True,
            )
            frac = pd.DataFrame(
                {"file": [Path(r.path).name for r in ok], "degraded fraction": [r.degraded_fraction for r in ok]}
            )
            st.plotly_chart(
                px.histogram(frac, x="degraded fraction", nbins=20, title="Per-file degraded fraction"),
                use_container_width=True,
            )

    # ---------------- gallery ----------------
    with tab_gallery:
        gal = EXAMPLES_DIR / "degradations"
        st.markdown(
            "One clean second of speech and the same second after each degradation (mel spectrogram, v1 rendering)."
        )
        if gal.exists():
            items = sorted(gal.glob("*.png"))
            cols = st.columns(4)
            for i, p in enumerate(items):
                with cols[i % 4]:
                    st.image(str(p), caption=p.stem, use_container_width=True)
                    wav = p.with_suffix(".flac")
                    if wav.exists():
                        st.audio(str(wav))
        else:
            st.info("examples/degradations not found")


@st.cache_data(show_spinner=False)
def _mel_image(path: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import librosa
    import librosa.display
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from aadd.features import load_audio

    y, sr = load_audio(path)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    fig = Figure(figsize=(12, 3), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(S, x_axis="time", y_axis="mel", sr=sr, ax=ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
