"""Record the report's file inspector while the playhead moves, as an animated GIF.

    aadd scan clips/ --model v3 --json scan.json
    aadd report scan.json -o report.html --audio embed
    python scripts/record_inspector_gif.py report.html clips/held_out_clip.wav docs/img/demo.gif [--dark]

The README demo is made this way, so it shows exactly the interface users get.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def record(report: str, file_key: str, out: str, dark: bool = False, fps: int = 5, width: int = 1120) -> None:
    frames = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": width, "height": 1000}, device_scale_factor=1, color_scheme="dark" if dark else "light")
        pg.goto("file://" + str(Path(report).resolve()))
        pg.wait_for_timeout(600)
        pg.fill("#find-in", file_key)
        pg.dispatch_event("#find-in", "change")
        pg.wait_for_function("document.querySelector('#mel image') !== null", timeout=30000)
        pg.add_style_tag(content="#modal{background:transparent!important;padding:0!important} #modal .panel{border-radius:0;border:none} .ivl:hover{border-color:var(--border)}")
        dur = pg.evaluate("document.getElementById('m-audio').duration")
        panel = pg.locator("#m-body")
        n = int(dur * fps) + 1
        for k in range(n + 2 * fps):  # hold the last frame for 2 s
            t = min(k / fps, dur - 0.01)
            pg.evaluate(f"document.getElementById('m-audio').currentTime={t}")
            pg.wait_for_timeout(60)
            img = Image.open(io.BytesIO(panel.screenshot())).convert("RGB")
            frames.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        b.close()
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, {Path(out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    record(sys.argv[1], sys.argv[2], sys.argv[3], dark="--dark" in sys.argv)
