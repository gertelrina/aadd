"""Decode every v3 audio source once into int16 memory maps.

Layout of ``<out>/``::

    <pool>.int16.npy   concatenated samples (16 kHz mono, int16)
    <pool>.index.json  [{"offset", "length", "split", "id", ...}, ...]

Pools: ``speech_en``, ``speech_ru``, ``noise``, ``music``, ``rir_sim``,
``rir_real``, ``rir_mit``. Splits are assigned deterministically (by speaker
for speech, by file / room hash for noises and RIRs) into train / val / test.
"""

from __future__ import annotations

import glob
import io
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR = 16000


def _split_by_key(key: str, val: float = 0.1, test: float = 0.1) -> str:
    h = (zlib.crc32(key.encode()) % 10000) / 10000
    return "test" if h < test else ("val" if h < test + val else "train")


def _load_file(path: str) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def _load_bytes(b: bytes) -> np.ndarray:
    import librosa
    import soundfile as sf

    y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def _to_int16(y: np.ndarray) -> np.ndarray:
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


def write_pool(out: Path, name: str, items: list[dict], loader, workers: int = 40) -> dict:
    """``items``: dicts with at least ``src`` (passed to ``loader``) and metadata."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{name}.int16.raw"
    index, offset = [], 0
    with open(tmp, "wb") as f, ProcessPoolExecutor(workers) as ex:
        for it, y in zip(items, ex.map(loader, [it["src"] for it in items], chunksize=16)):
            if y is None or len(y) < SR // 4:
                continue
            a = _to_int16(y)
            f.write(a.tobytes())
            meta = {k: v for k, v in it.items() if k != "src" or isinstance(v, str)}
            if not isinstance(meta.get("src"), str):
                meta.pop("src", None)
            meta.update(offset=offset, length=len(a))
            index.append(meta)
            offset += len(a)
    arr = np.memmap(tmp, dtype=np.int16, mode="r", shape=(offset,))
    np.save(out / f"{name}.int16.npy", arr)
    del arr
    tmp.unlink()
    (out / f"{name}.index.json").write_text(json.dumps(index))
    hours = offset / SR / 3600
    by_split = {}
    for m in index:
        by_split[m["split"]] = by_split.get(m["split"], 0) + m["length"] / SR / 3600
    return {
        "pool": name,
        "items": len(index),
        "hours": round(hours, 2),
        "hours_by_split": {k: round(v, 2) for k, v in by_split.items()},
    }


class Pool:
    """Read-only view of a pool; ``pool[i]`` -> float32 array; ``pool.split(s)`` -> indices."""

    def __init__(self, root: str | os.PathLike, name: str):
        root = Path(root)
        self.data = np.load(root / f"{name}.int16.npy", mmap_mode="r")
        self.index = json.loads((root / f"{name}.index.json").read_text())
        self.name = name

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int) -> np.ndarray:
        m = self.index[i]
        return self.data[m["offset"] : m["offset"] + m["length"]].astype(np.float32) / 32767.0

    def split(self, s: str) -> list[int]:
        return [i for i, m in enumerate(self.index) if m["split"] == s]

    def view(self, s: str) -> PoolView:
        return PoolView(self, self.split(s))


class PoolView:
    def __init__(self, pool: Pool, idx: list[int]):
        self.pool, self.idx = pool, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.pool[self.idx[i]]

    def meta(self, i: int) -> dict:
        return self.pool.index[self.idx[i]]


# --------------------------------------------------------------------------- #
def build_all(v3_dir: str, libri_dir: str, out: str, workers: int = 40) -> list[dict]:
    v3, out_p = Path(v3_dir), Path(out)
    reports = []

    # English speech: train-clean-100 / dev-clean / test-clean
    items = []
    for split, d in [
        ("train", v3 / "LibriSpeech/train-clean-100"),
        ("val", Path(libri_dir) / "dev-clean"),
        ("test", Path(libri_dir) / "test-clean"),
    ]:
        for p in sorted(glob.glob(str(d / "*/*/*.flac"))):
            spk = Path(p).stem.split("-")[0]
            items.append({"src": p, "split": split, "id": Path(p).stem, "speaker": spk, "lang": "en"})
    reports.append(write_pool(out_p, "speech_en", items, _load_file, workers))

    # Russian speech: RuLS parquet
    import pyarrow.parquet as pq

    items = []
    rdir = v3 / "hf/istupakov__russian_librispeech/data"
    for f in sorted(rdir.glob("*.parquet")):
        split = {"train": "train", "validation": "val", "test": "test"}[f.name.split("-")[0]]
        t = pq.read_table(f, columns=["audio", "audio_filepath"])
        for a, fp in zip(t.column("audio").to_pylist(), t.column("audio_filepath").to_pylist()):
            parts = fp.split("/")
            items.append(
                {
                    "src": a["bytes"],
                    "split": split,
                    "id": fp,
                    "speaker": parts[1] if len(parts) > 2 else "?",
                    "lang": "ru",
                }
            )
    reports.append(write_pool(out_p, "speech_ru", items, _load_bytes, workers))
    del items

    # MUSAN noise and music, split by file
    for name, sub in [("noise", "noise"), ("music", "music")]:
        files = sorted(glob.glob(str(v3 / f"musan/{sub}/*/*.wav")))
        items = [{"src": p, "split": _split_by_key(Path(p).name), "id": str(Path(p).relative_to(v3))} for p in files]
        reports.append(write_pool(out_p, name, items, _load_file, workers))

    # RIRs: simulated (split by room), real (test only), MIT survey (ood)
    items = []
    for size in ["smallroom", "mediumroom", "largeroom"]:
        for line in (v3 / f"RIRS_NOISES/simulated_rirs/{size}/rir_list").read_text().splitlines():
            parts = line.split()
            room, rel = parts[3], parts[4]
            items.append(
                {"src": str(v3 / rel), "split": _split_by_key(room), "id": parts[1], "room": room, "size": size}
            )
    reports.append(write_pool(out_p, "rir_sim", items, _load_file, workers))
    items = []
    for line in (v3 / "RIRS_NOISES/real_rirs_isotropic_noises/rir_list").read_text().splitlines():
        parts = line.split()
        items.append({"src": str(v3 / parts[4]), "split": "test", "id": parts[1], "room": parts[3]})
    reports.append(write_pool(out_p, "rir_real", items, _load_file, workers))
    mit = sorted(glob.glob(str(v3 / "hf/TaterTotterson__MIT_environmental_impulse_responses/16khz/*.wav")))
    reports.append(
        write_pool(
            out_p, "rir_mit", [{"src": p, "split": "test", "id": Path(p).stem} for p in mit], _load_file, workers
        )
    )
    (out_p / "sources_report.json").write_text(json.dumps(reports, indent=1))
    return reports


if __name__ == "__main__":
    import sys

    for r in build_all(sys.argv[1], sys.argv[2], sys.argv[3], workers=int(sys.argv[4]) if len(sys.argv) > 4 else 40):
        print(r, flush=True)
