"""On-the-fly training data and fixed evaluation sets for v3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aadd.v3.sources import Pool
from aadd.v3.synth import CLASSES_V3, Assets, Synthesizer


class Pools:
    """All memmapped pools of a build directory (opened lazily, safe to fork)."""

    def __init__(self, root: str):
        self.root = root
        self._p: dict[str, Pool] = {}

    def __getitem__(self, name: str) -> Pool:
        if name not in self._p:
            self._p[name] = Pool(self.root, name)
        return self._p[name]

    def assets(self, split: str, rir: str = "sim") -> Assets:
        if rir == "sim":
            rirs = self["rir_sim"].view(split)
        elif rir == "real":
            rirs = self["rir_real"].view("test")
        else:
            rirs = self["rir_mit"].view("test")
        return Assets(noises=self["noise"].view(split), music=self["music"].view(split), rirs=rirs)


class SynthDataset(Dataset):
    """Infinite-style map dataset: item ``i`` is a fresh random window (seeded by ``seed`` and ``i``)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        langs=("en", "ru"),
        seconds: float = 8.0,
        length: int = 10**9,
        seed: int = 0,
        rir: str = "sim",
    ):
        self.root, self.split, self.langs, self.seconds = root, split, list(langs), seconds
        self.length, self.seed, self.rir = length, seed, rir
        self._init = False

    def _lazy(self):
        if self._init:
            return
        self.pools = Pools(self.root)
        self.speech = {lang: self.pools[f"speech_{lang}"].view(self.split) for lang in self.langs}
        self.syn = Synthesizer(self.pools.assets(self.split, self.rir), seconds=self.seconds)
        self._init = True

    def __len__(self):
        return self.length

    def sample(self, i: int):
        self._lazy()
        rng = np.random.default_rng([self.seed, i])
        lang = self.langs[int(rng.integers(len(self.langs)))]
        sp = self.speech[lang]
        x, y, ig, ev = self.syn.make(sp[int(rng.integers(len(sp)))], rng)
        return x, y, ig, lang, ev

    def __getitem__(self, i: int):
        x, y, ig, _lang, _ = self.sample(i)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(ig)


def make_fixed_set(
    root: str, out: str, split: str, lang: str, n: int, seed: int, rir: str = "sim", workers: int = 40
) -> dict:
    """Generate ``n`` windows once and store them (int16 audio + targets + events)."""
    from concurrent.futures import ProcessPoolExecutor

    key = (root, split, lang, seed, rir)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_sample_job, [(key, i) for i in range(n)], chunksize=8))
    X = np.stack([(np.clip(r[0], -1, 1) * 32767).astype(np.int16) for r in res])
    Y = np.stack([r[1] for r in res]).astype(np.uint8)
    IG = np.stack([r[2] for r in res])
    events = [
        [{"cls": e.cls, "start": e.start, "end": e.end, "params": _jsonable(e.params)} for e in r[4]] for r in res
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, Y=Y, IG=IG, classes=np.array(CLASSES_V3))
    Path(out).with_suffix(".events.json").write_text(json.dumps(events))
    return {"path": out, "n": n, "pos_frames_per_class": dict(zip(CLASSES_V3, Y.sum((0, 1)).tolist()))}


_DS_CACHE: dict = {}


def _sample_job(args):
    key, i = args
    if key not in _DS_CACHE:  # one dataset (and one set of opened pools) per worker process
        root, split, lang, seed, rir = key
        _DS_CACHE[key] = SynthDataset(root, split, langs=(lang,), seed=seed, rir=rir)
    return _DS_CACHE[key].sample(i)


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


class FixedSet(Dataset):
    def __init__(self, path: str):
        z = np.load(path)
        self.X, self.Y, self.IG = z["X"], z["Y"], z["IG"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i].astype(np.float32) / 32767.0),
            torch.from_numpy(self.Y[i].astype(np.float32)),
            torch.from_numpy(self.IG[i]),
        )
