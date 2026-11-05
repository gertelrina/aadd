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


def test_build_report_contains_sections():
    classes = ["noise", "reverb", "codec"]
    rng = np.random.default_rng(0)
    files = [_file(f"/d/sub{i % 2}/f{i}.wav", rng.random((5, 3)), classes) for i in range(6)]
    files.append({"path": "/d/sub0/bad.wav", "duration_s": 0, "file": {"status": "error"}, "segments": []})
    html = build_report({"model": "v3", "threshold": 0.5, "classes": classes, "files": files}, title="T")
    for needle in [
        "files analysed",
        "Share of audio time affected",
        "Degradations that occur together",
        "By folder",
        "Most degraded files",
        "sub0",
        "sub1",
        "Background noise",
    ]:
        assert needle in html
    assert html.startswith("<!doctype html>") and "http" not in html.split("<body>")[1].split("</style>")[0]
