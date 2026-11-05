"""Generate the fixed v3 validation / test sets (seeded, stored once)."""

import json
import sys
from pathlib import Path

from aadd.v3.data import make_fixed_set

SETS = [
    # name, split, lang, n, seed, rir
    ("val_en", "val", "en", 1000, 101, "sim"),
    ("val_ru", "val", "ru", 1000, 102, "sim"),
    ("test_en", "test", "en", 3000, 201, "sim"),
    ("test_ru", "test", "ru", 3000, 202, "sim"),
    ("test_en_real_rir", "test", "en", 1000, 203, "real"),
    ("test_en_mit_rir", "test", "en", 1000, 204, "mit"),
]

if __name__ == "__main__":
    pools, out = sys.argv[1], Path(sys.argv[2])
    rep = {}
    for name, split, lang, n, seed, rir in SETS:
        rep[name] = make_fixed_set(pools, str(out / f"{name}.npz"), split, lang, n, seed, rir=rir, workers=44)
        print(name, rep[name], flush=True)
    (out / "sets_report.json").write_text(json.dumps(rep, indent=1))
