import base64
import json
import re

import numpy as np

from aadd.report import build_report


def _file(path, probs, classes):
    segs = [
        {"path": path, "start_s": float(i), "end_s": float(i + 1), **{f"p_{c}": float(p) for c, p in zip(classes, row)}}
        for i, row in enumerate(probs)
    ]
    return {
        "path": path,
        "duration_s": float(len(probs)),
        "file": {"status": "ok", "duration_s": float(len(probs))},
        "segments": segs,
    }


def test_build_report_embeds_data_and_controls():
    classes = ["noise", "reverb", "codec"]
    probs = [np.random.default_rng(i).random((5, 3)) for i in range(6)]
    files = [_file(f"/d/sub{i % 2}/f{i}.wav", p, classes) for i, p in enumerate(probs)]
    files.append({"path": "/d/sub0/bad.wav", "duration_s": 0, "file": {"status": "error"}, "segments": []})
    html = build_report({"model": "v3", "threshold": 0.5, "classes": classes, "files": files}, title="T")
    assert html.startswith("<!doctype html>")
    for needle in [
        'id="thr"',
        'id="tol"',
        'id="chips"',
        'id="dl"',
        "Clean audio kept vs threshold",
        "Most degraded files",
    ]:
        assert needle in html
    data = json.loads(
        re.search(r'<script id="aadd-data" type="application/json">(.*?)</script>', html, re.DOTALL).group(1)
    )
    assert data["errors"] == 1 and len(data["files"]) == 6 and data["pretty"][0] == "Background noise"
    q = np.frombuffer(base64.b64decode(data["files"][0]["q"]), dtype=np.uint8).reshape(5, 3)
    assert np.abs(q / 255 - probs[0]).max() <= 0.5 / 255 + 1e-9
    assert {f["s"] for f in data["files"]} == {"sub0", "sub1"}
