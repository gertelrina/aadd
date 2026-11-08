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
    data = json.loads(re.search(r"<script>window.AADD_DATA=(.*?);</script>", html, re.DOTALL).group(1))
    assert data["errors"] == 1 and len(data["files"]) == 6 and data["pretty"][0] == "Background noise"
    q = np.frombuffer(base64.b64decode(data["files"][0]["q"]), dtype=np.uint8).reshape(5, 3)
    assert np.abs(q / 255 - probs[0]).max() <= 0.5 / 255 + 1e-9
    assert {f["s"] for f in data["files"]} == {"sub0", "sub1"}


def test_build_compare_pairs_files():
    from aadd.compare import build_compare

    classes = ["noise", "reverb", "codec"]
    rng = np.random.default_rng(1)
    before = {
        "model": "v3",
        "threshold": 0.5,
        "classes": classes,
        "files": [_file(f"/raw/s/f{i}.wav", rng.random((4, 3)), classes) for i in range(4)],
    }
    after = {
        "model": "v3",
        "threshold": 0.5,
        "classes": classes,
        "files": [_file(f"/clean/s/f{i}.wav", rng.random((4, 3)) * 0.3, classes) for i in range(4)],
    }
    html = build_compare(before, after, audio="none")
    assert "window.AADD_BEFORE=" in html and "window.AADD_AFTER=" in html
    for needle in [
        "Every degradation, before and after",
        "Most improved files",
        "Files with new problems",
        "function inspector(",
    ]:
        assert needle in html


def test_nisqa_scores_reach_csv_json_and_compare():
    from aadd.compare import build_compare
    from aadd.infer import FileResult

    classes = ["noise", "reverb"]
    r = FileResult(
        "/x/a.wav", 16000, 2.0, np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.zeros((2, 2)), classes, 0.5, "v3"
    )
    r.nisqa = {"mos": 3.5, "noisiness": 3.0, "discontinuity": 4.0, "colouration": 4.1, "loudness": 4.2}
    assert r.file_row()["nisqa_mos"] == 3.5 and r.to_dict()["nisqa"]["loudness"] == 4.2

    def scan(root, mos):
        fs = [_file(f"/{root}/s/f{i}.wav", np.full((3, 2), 0.1), classes) for i in range(3)]
        for f, m in zip(fs, mos):
            f["nisqa"] = {"mos": m, "noisiness": m, "discontinuity": m, "colouration": m, "loudness": m}
        return {"model": "v3", "threshold": 0.5, "classes": classes, "files": fs}

    html = build_compare(scan("raw", [2.0, 2.5, float("nan")]), scan("clean", [4.0, 4.5, 3.0]), audio="none")
    before = json.loads(re.search(r"window.AADD_BEFORE=(.*?);window.AADD_AFTER", html, re.DOTALL).group(1))
    assert [f.get("m", [None])[0] for f in before["files"]] == [2.0, 2.5, None]  # NaN dropped
    assert "NISQA MOS" in html
