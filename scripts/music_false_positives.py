"""Music detections where no music is present: v3.0 vs v3.1 on VoiceBank-DEMAND (by DEMAND noise
type) and on the DNS-2020 synthetic dev-test. Writes runs/music_false_positives.json.

    python scripts/music_false_positives.py --hf ood_hf --vb-log log_testset.txt --out runs/music_false_positives.json

``log_testset.txt`` is the VoiceBank-DEMAND test log (file, noise type, SNR) from
https://datashare.ed.ac.uk/handle/10283/2791 (logfiles.zip). A clip counts as a music detection when the
95th percentile of its music frame probabilities is >= 0.5 (the clip-score rule of the benchmark tables).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    import pyarrow.parquet as pq

    from aadd.model import resolve_model_path
    from aadd.v3.evaluate import _decode
    from aadd.v3.infer import FrameModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--vb-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="v3.0,v3")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    noise_type = {}
    for line in Path(a.vb_log).read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            noise_type[parts[0]] = parts[1]
    vb = pq.read_table(glob.glob(f"{a.hf}/JacobLinCool__VoiceBank-DEMAND-16k/data/test*.parquet")[0]).to_pylist()
    dns_dir = f"{a.hf}/nkdem__DNS-Challenge-2020-DevTest-16k/data"
    dns = (
        pq.read_table(glob.glob(f"{dns_dir}/synthetic_no_reverb*")[0]).to_pylist()
        + pq.read_table(glob.glob(f"{dns_dir}/synthetic_with_reverb*")[0]).to_pylist()
    )

    rep = {"rule": "95th percentile of the music frame probability >= 0.5", "models": {}}
    for name in a.models.split(","):
        fm = FrameModel(str(resolve_model_path(name)), a.device or "cpu")
        j = fm.classes.index("music")

        def det(y, fm=fm, j=j):
            return float(np.percentile(fm.frame_probs(y)[:, j], 95)) >= 0.5

        vb_noisy = {r["id"]: det(_decode(r["noisy"])) for r in vb}
        vb_clean = {r["id"]: det(_decode(r["clean"])) for r in vb}
        by_type = {}
        for k, v in vb_noisy.items():
            by_type.setdefault(noise_type.get(k, "unknown"), []).append(v)
        r = {
            "voicebank_noisy_by_noise_type": {
                t: {"n": len(v), "share": float(np.mean(v))} for t, v in sorted(by_type.items())
            },
            "voicebank_noisy_all": {"n": len(vb_noisy), "share": float(np.mean(list(vb_noisy.values())))},
            "voicebank_clean_all": {"n": len(vb_clean), "share": float(np.mean(list(vb_clean.values())))},
            "voicebank_all_clips": {
                "n": 2 * len(vb),
                "share": float(np.mean(list(vb_noisy.values()) + list(vb_clean.values()))),
            },
        }
        dn = [det(_decode(x["noisy"])) for x in dns]
        dc = [det(_decode(x["clean"])) for x in dns]
        r["dns_noisy"] = {"n": len(dn), "share": float(np.mean(dn))}
        r["dns_clean"] = {"n": len(dc), "share": float(np.mean(dc))}
        r["dns_all_clips"] = {"n": len(dn) + len(dc), "share": float(np.mean(dn + dc))}
        rep["models"][name] = r
        print(name, json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
