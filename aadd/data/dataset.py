"""Datasets built from a ``make-data`` directory (``audio/*.flac`` + ``labels/*.csv``).

* :func:`load_manifest` – list (audio, labels, source, copy) rows.
* :func:`labels_to_targets` – CSV rows -> [n_chunks, C] float targets (single-hot per chunk).
* :func:`featurize_dir` – precompute features for every chunk into ``.npz``
  (X float16 [N, C, F, T], Y uint8 [N, classes], plus an index) for fast training.
* :class:`ChunkArrayDataset` – torch Dataset over such an ``.npz``.
"""

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aadd.constants import CLASSES, NONE_CLASS
from aadd.data.augment import read_labels
from aadd.features import FeatureSpec, load_audio, segment_audio


def load_manifest(data_dir: str | os.PathLike) -> list[dict]:
    data_dir = Path(data_dir)
    mf = data_dir / "manifest.csv"
    if mf.exists():
        with open(mf, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["copy"] = int(r["copy"])
        return rows
    # fall back to pairing by stem (works for the original v1 layout too)
    rows = []
    for a in sorted((data_dir / "audio").glob("*.flac")):
        lab = data_dir / "labels" / (a.stem + ".csv")
        if lab.exists():
            stem = a.stem
            copy = int(stem.rsplit("_", 1)[1]) if "_" in stem and stem.rsplit("_", 1)[1].isdigit() else -1
            rows.append({"audio": str(a), "labels": str(lab), "source": stem.rsplit("_", 1)[0], "copy": copy})
    return rows


def speaker_of(row: dict) -> str:
    """LibriSpeech file names are ``<speaker>-<chapter>-<utt>``."""
    return Path(row["source"]).stem.split("-")[0]


def labels_to_targets(rows: list[dict], classes: list[str] = CLASSES) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(rows), len(classes)), dtype=np.float32)
    for i, r in enumerate(rows):
        name = (r.get("augmentation") or "None").strip()
        if name in ("", "nan"):
            name = NONE_CLASS
        y[i, idx[name]] = 1.0
    return y


def _featurize_one(args):
    audio, labels, spec_dict = args
    spec = FeatureSpec.from_dict(spec_dict)
    y, sr = load_audio(audio, sr=spec.sample_rate)
    segs = segment_audio(y, sr, spec.segment_seconds, keep_tail=False)
    rows = read_labels(labels)
    n = min(len(segs), len(rows))
    if n == 0:
        return None
    X = np.stack([spec(s.samples, sr) for s in segs[:n]]).astype(np.float16)
    Y = labels_to_targets(rows[:n]).astype(np.uint8)
    return audio, X, Y


def featurize_dir(
    data_dir: str | os.PathLike,
    out_path: str | os.PathLike,
    spec: FeatureSpec,
    workers: int | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    """Precompute chunk features for a whole make-data directory into one ``.npz``."""
    rows = load_manifest(data_dir)
    if limit:
        rows = rows[:limit]
    jobs = [(r["audio"], r["labels"], spec.to_dict()) for r in rows]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    Xs, Ys, files, file_idx, chunk_idx = [], [], [], [], []
    it = None
    if progress:
        from tqdm import tqdm

        it = tqdm(total=len(jobs), unit="file", desc="featurize")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_featurize_one, jobs, chunksize=8):
            if it:
                it.update(1)
            if res is None:
                continue
            audio, X, Y = res
            fi = len(files)
            files.append(audio)
            Xs.append(X)
            Ys.append(Y)
            file_idx.append(np.full(len(X), fi, dtype=np.int32))
            chunk_idx.append(np.arange(len(X), dtype=np.int32))
    if it:
        it.close()
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    out = {
        "X": X,
        "Y": Y,
        "file_idx": np.concatenate(file_idx),
        "chunk_idx": np.concatenate(chunk_idx),
        "files": np.array(files),
        "classes": np.array(CLASSES),
        "spec": np.array([__import__("json").dumps(spec.to_dict())]),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    return {"n_chunks": int(len(X)), "n_files": len(files), "shape": list(X.shape[1:]), "path": str(out_path)}


class ChunkArrayDataset(Dataset):
    """Dataset over a featurized ``.npz`` (optionally restricted to a subset of rows)."""

    def __init__(self, npz_path: str | os.PathLike, indices: np.ndarray | None = None, augment=None):
        z = np.load(npz_path, allow_pickle=False)
        self.X = z["X"]
        self.Y = z["Y"]
        self.file_idx = z["file_idx"]
        self.files = z["files"]
        self.classes = list(z["classes"])
        self.spec = FeatureSpec.from_dict(__import__("json").loads(str(z["spec"][0])))
        self.indices = np.arange(len(self.X)) if indices is None else np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x = torch.from_numpy(self.X[j].astype(np.float32))
        if self.augment is not None:
            x = self.augment(x)
        return x, torch.from_numpy(self.Y[j].astype(np.float32))

    def speakers(self) -> np.ndarray:
        """Speaker id for every row (LibriSpeech naming), for leakage-free splits."""
        sp = np.array([Path(f).stem.rsplit("_", 1)[0].split("-")[0] for f in self.files])
        return sp[self.file_idx]


def split_by_speaker(ds: ChunkArrayDataset, val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for (train, val) with disjoint speakers."""
    sp = ds.speakers()
    uniq = np.unique(sp)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val_sp = set(uniq[:n_val].tolist())
    mask = np.array([s in val_sp for s in sp])
    return np.where(~mask)[0], np.where(mask)[0]


class SpecMask:
    """Light SpecAugment-style masking for 1xFxT inputs (training only)."""

    def __init__(self, freq: int = 12, time: int = 8, p: float = 0.5):
        self.freq, self.time, self.p = freq, time, p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        x = x.clone()
        _, F, T = x.shape
        f = int(torch.randint(0, self.freq + 1, (1,)))
        f0 = int(torch.randint(0, max(1, F - f), (1,)))
        t = int(torch.randint(0, self.time + 1, (1,)))
        t0 = int(torch.randint(0, max(1, T - t), (1,)))
        x[:, f0 : f0 + f, :] = 0.0
        x[:, :, t0 : t0 + t] = 0.0
        return x
