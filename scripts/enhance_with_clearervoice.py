"""Enhance only the files AADD flagged, with ClearerVoice-Studio (Apache-2.0); copy the rest unchanged.

    pip install clearvoice            # in a separate environment: it pins numpy<2, soundfile 0.12, opencv
    aadd scan raw/ --model v3 --json before.json
    python scripts/enhance_with_clearervoice.py before.json raw/ enhanced/ [--model MossFormerGAN_SE_16K] [--classes noise,music]
    aadd scan enhanced/ --model v3 --json after.json
    aadd compare before.json after.json -o compare.html --audio embed

By default only files with background noise or music are enhanced: a speech-enhancement
model does not remove reverberation and, on reverberant speech, adds band-limiting and
codec-like artefacts (see README -> Before / after).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_json")
    ap.add_argument("src_root")
    ap.add_argument("dst_root")
    ap.add_argument("--model", default="MossFormerGAN_SE_16K")
    ap.add_argument("--classes", default="noise,music", help="AADD classes that trigger enhancement")
    ap.add_argument("--min-fraction", type=float, default=0.1, help="minimum share of degraded seconds")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index for ClearerVoice (it otherwise picks one itself)")
    a = ap.parse_args()

    src, dst = Path(a.src_root).resolve(), Path(a.dst_root).resolve()
    fix = set(a.classes.split(","))
    scan = json.loads(Path(a.scan_json).read_text())
    flag, keep = [], []
    for f in scan["files"]:
        det = set(f["file"]["detected"].split(";")) - {""}
        (flag if det & fix and f["file"]["degraded_fraction"] >= a.min_fraction else keep).append(
            Path(f["path"]).resolve()
        )
    print(f"{len(flag)} files to enhance with {a.model}, {len(keep)} copied unchanged", flush=True)
    for p in keep:
        d = dst / p.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)

    import clearvoice.networks as net

    for obj in vars(net).values():  # pin the GPU instead of ClearerVoice's "freest GPU" heuristic
        if isinstance(obj, type) and hasattr(obj, "get_free_gpu"):
            obj.get_free_gpu = lambda self, g=a.gpu: g
    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_enhancement", model_names=[a.model])
    t0 = time.time()
    for i, p in enumerate(flag):
        d = dst / p.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        cv.write(cv(input_path=str(p), online_write=False), output_path=str(d))
        if i % 200 == 0:
            print(f"{i}/{len(flag)} {time.time() - t0:.0f}s", flush=True)
    (dst / "enhanced_files.json").write_text(
        json.dumps({"model": a.model, "classes": sorted(fix), "enhanced": [str(p) for p in flag]}, indent=0)
    )
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
