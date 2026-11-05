"""Render the README demo GIF from a demo .npz (audio, frame probs, targets).

python scripts/make_demo_gif.py demo.npz docs/img/demo.gif
"""

from __future__ import annotations

import sys

import librosa
import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

SR, FRAME = 16000, 320
LABELS = {
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
}
BG, FG, MUTED, ACCENT = "#0f1117", "#e6e6e6", "#8a8f98", "#ffcc33"


def main(src: str, dst: str, fps: int = 5):
    z = np.load(src, allow_pickle=False)
    x, P, Y = z["audio"], z["probs"], z["target"]
    classes = [str(c) for c in z["classes"]]
    dur = len(x) / SR
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=160, n_mels=96), ref=np.max
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(10, 6.4), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.3], hspace=0.08, left=0.2, right=0.97, top=0.88, bottom=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    for ax in (ax0, ax1):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=MUTED, length=0)

    fig.text(0.2, 0.95, "AADD v3", color=ACCENT, fontsize=16, fontweight="bold", va="center")
    fig.text(0.31, 0.95, "which degradation, and where — every 20 ms", color=FG, fontsize=13, va="center")
    fig.text(
        0.2, 0.915, "dashed boxes = ground truth   ·   colour = model probability", color=MUTED, fontsize=9, va="center"
    )

    ax0.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, dur, 0, S.shape[0]])
    ax0.set_yticks([])
    ax0.set_ylabel("mel\nspectrogram", color=MUTED, rotation=0, ha="right", va="center")
    plt.setp(ax0.get_xticklabels(), visible=False)

    order = list(range(len(classes)))
    ax1.imshow(
        P[:, order].T,
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
        extent=[0, dur, len(order) - 0.5, -0.5],
        interpolation="nearest",
    )
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([LABELS.get(classes[j], classes[j]) for j in order], color=FG)
    ax1.set_xlabel("seconds", color=MUTED)
    for r, j in enumerate(order):
        on = Y[:, j] > 0
        if not on.any():
            continue
        # merge gaps shorter than 0.4 s so sparse classes (clicks, clipping) show as one event
        idx = np.flatnonzero(on)
        filled = on.copy()
        for a, b in itertools.pairwise(idx):
            if b - a < int(0.4 * SR / FRAME):
                filled[a:b] = True
        edges = np.flatnonzero(np.diff(np.r_[0, filled.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax1.add_patch(
                Rectangle(
                    (a * FRAME / SR, r - 0.45),
                    (b - a) * FRAME / SR,
                    0.9,
                    fill=False,
                    ec="#ffffff",
                    lw=1.4,
                    ls=(0, (3, 2)),
                    zorder=4,
                )
            )
            ax1.get_yticklabels()[r].set_color(ACCENT)
    for r in range(len(order) + 1):
        ax1.axhline(r - 0.5, color="#22252e", lw=0.6)

    covers = [ax.add_patch(Rectangle((0, -1e3), dur, 2e6, color=BG, alpha=0.78, zorder=5)) for ax in (ax0, ax1)]
    lines = [ax.axvline(0, color=ACCENT, lw=1.6, zorder=6) for ax in (ax0, ax1)]
    clock = fig.text(0.97, 0.95, "", color=MUTED, fontsize=11, ha="right", va="center", family="DejaVu Sans Mono")
    ax1.set_xlim(0, dur)

    frames = []
    n = int(dur * fps) + 1
    for k in range(n + fps * 2):  # hold the last frame for 2 s
        t = min(k / fps, dur)
        for c in covers:
            c.set_x(t)
            c.set_width(max(dur - t, 0))
        for ln in lines:
            ln.set_xdata([t, t])
            ln.set_visible(t < dur)
        clock.set_text(f"{t:5.1f} / {dur:.1f} s")
        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(img.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    frames[-1].convert("RGB").save(dst.replace(".gif", "_final.png"))
    print(f"wrote {dst}: {len(frames)} frames")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
