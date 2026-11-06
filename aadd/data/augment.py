"""Build the synthetic degradation dataset used for v1 and v2.

Protocol:

1. every clean file is cut into non-overlapping 1-s chunks (the trailing
   partial second is dropped);
2. each chunk independently gets, with probability 0.5, ONE degradation
   drawn uniformly from the seven audiomentations transforms in
   :data:`aadd.constants.AUGMENT_PARAMS`, otherwise it stays clean;
3. the chunks are concatenated back; a CSV next to the audio records, per
   chunk, ``timestamp_start, timestamp_end, augmentation, parameters, sample_rate``;
4. for every source file ``copies`` degraded versions are written
   (``<name>_1.flac`` ... ``<name>_<copies>.flac``) plus the untouched original
   as ``<name>_0.flac`` with an all-``None`` CSV.

Every transform keeps the chunk length (audiomentations ``TimeStretch`` crops /
zero-pads with ``leave_length_unchanged=True``; ``Shift`` rolls over), so the
per-chunk labels stay aligned with time.

Reproducibility: the RNG is seeded per (source file, copy index) from ``seed``,
so the dataset is identical regardless of worker count or ordering.
"""

from __future__ import annotations

import csv
import json
import os
import random
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from aadd.constants import AUGMENT_PARAMS, AUGMENT_PROBABILITY, DEGRADATION_CLASSES, SAMPLE_RATE, SEGMENT_SECONDS

CSV_FIELDS = ["timestamp_start", "timestamp_end", "augmentation", "parameters", "sample_rate"]


def build_transforms(stretch_method: str = "librosa_phase_vocoder") -> dict:
    """One audiomentations transform per class, each with p=1.0.

    ``stretch_method``: audiomentations >= 0.39 defaults to ``signalsmith_stretch``
    for PitchShift / TimeStretch; v1 data was made with 0.38, which only had the
    librosa phase vocoder, so that is the default here.
    """
    import audiomentations as A

    P = AUGMENT_PARAMS
    t = {
        "AddGaussianNoise": A.AddGaussianNoise(**P["AddGaussianNoise"], p=1.0),
        "ClippingDistortion": A.ClippingDistortion(**P["ClippingDistortion"], p=1.0),
        "LowPassFilter": A.LowPassFilter(**P["LowPassFilter"], p=1.0),
        "HighPassFilter": A.HighPassFilter(**P["HighPassFilter"], p=1.0),
        "Shift": A.Shift(**P["Shift"], p=1.0),
    }
    try:
        t["PitchShift"] = A.PitchShift(**P["PitchShift"], method=stretch_method, p=1.0)
        t["TimeStretch"] = A.TimeStretch(**P["TimeStretch"], leave_length_unchanged=True, method=stretch_method, p=1.0)
    except TypeError:  # audiomentations < 0.39: no `method` argument, librosa only
        t["PitchShift"] = A.PitchShift(**P["PitchShift"], p=1.0)
        t["TimeStretch"] = A.TimeStretch(**P["TimeStretch"], leave_length_unchanged=True, p=1.0)
    assert set(t) == set(DEGRADATION_CLASSES)
    return t


def _json_safe(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def degrade_signal(
    y: np.ndarray,
    sr: int,
    transforms: dict,
    rng: random.Random,
    p_aug: float = AUGMENT_PROBABILITY,
    segment_seconds: float = SEGMENT_SECONDS,
) -> tuple[np.ndarray, list[dict]]:
    """Apply the per-chunk protocol to one signal. Returns (audio, csv rows)."""
    n = round(sr * segment_seconds)
    n_chunks = len(y) // n
    names = sorted(transforms)
    out, rows = [], []
    for i in range(n_chunks):
        chunk = y[i * n : (i + 1) * n].astype(np.float32)
        if rng.random() < p_aug:
            name = names[rng.randrange(len(names))]
            t = transforms[name]
            # audiomentations draws its own parameters from `random`/`np.random`,
            # which the caller seeds; p=1.0 so should_apply is always True.
            # A handful of parameter draws crash inside audiomentations (e.g. Shift
            # by exactly +-100 % leaves an empty fade slice); redraw up to 3 times.
            src = chunk
            for attempt in range(3):
                try:
                    chunk = t(samples=src, sample_rate=sr)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"{name} failed 3 times on chunk {i}: {e}") from e
            params = _json_safe(dict(t.parameters))
            if len(chunk) != n:  # defensive; all configured transforms keep length
                chunk = np.pad(chunk[:n], (0, max(0, n - len(chunk))))
        else:
            name, params = "None", {}
        out.append(np.clip(chunk, -1.0, 1.0))
        rows.append(
            {
                "timestamp_start": i,
                "timestamp_end": i + 1,
                "augmentation": name,
                "parameters": json.dumps(params) if params else "",
                "sample_rate": sr,
            }
        )
    audio = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    return audio, rows


def write_labels(path: str | os.PathLike, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_labels(path: str | os.PathLike) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class Job:
    src: str
    out_audio: str
    out_csv: str
    copy: int
    seed: int
    p_aug: float
    stretch_method: str


def _file_seed(seed: int, src: str, copy: int) -> int:
    return (seed * 1_000_003 + zlib.crc32(f"{Path(src).name}|{copy}".encode())) % (2**31 - 1)


def _run_job(job: Job) -> dict:
    try:
        return _run_job_inner(job)
    except Exception as e:  # noqa: BLE001 - one bad file must not kill a 16k-file run
        return {"src": job.src, "copy": job.copy, "skipped": f"{type(e).__name__}: {e}"}


def _run_job_inner(job: Job) -> dict:
    import librosa

    s = _file_seed(job.seed, job.src, job.copy)
    rng = random.Random(s)
    random.seed(s)
    np.random.seed(s % (2**32 - 1))
    y, sr = librosa.load(job.src, sr=SAMPLE_RATE, mono=True)
    if job.copy == 0:
        n = round(sr * SEGMENT_SECONDS)
        n_chunks = len(y) // n
        audio = y[: n_chunks * n]
        rows = [
            {"timestamp_start": i, "timestamp_end": i + 1, "augmentation": "None", "parameters": "", "sample_rate": sr}
            for i in range(n_chunks)
        ]
    else:
        transforms = build_transforms(job.stretch_method)
        audio, rows = degrade_signal(y, sr, transforms, rng, p_aug=job.p_aug)
    if len(rows) == 0:
        return {"src": job.src, "copy": job.copy, "skipped": "shorter than 1 s"}
    sf.write(job.out_audio, audio, sr, subtype="PCM_16")
    write_labels(job.out_csv, rows)
    counts = {}
    for r in rows:
        counts[r["augmentation"]] = counts.get(r["augmentation"], 0) + 1
    return {
        "src": job.src,
        "copy": job.copy,
        "audio": job.out_audio,
        "labels": job.out_csv,
        "n_chunks": len(rows),
        "n_degraded": sum(v for k, v in counts.items() if k != "None"),
        "counts": counts,
    }


def list_librispeech(root: str | os.PathLike) -> list[str]:
    """All .flac under a LibriSpeech split directory (speaker/chapter/*.flac)."""
    return sorted(str(p) for p in Path(root).rglob("*.flac"))


def make_dataset(
    sources: list[str],
    out_dir: str | os.PathLike,
    copies: int = 5,
    seed: int = 0,
    p_aug: float = AUGMENT_PROBABILITY,
    stretch_method: str = "librosa_phase_vocoder",
    workers: int | None = None,
    progress: bool = True,
    limit: int | None = None,
) -> list[dict]:
    """Write ``<out_dir>/audio/*.flac``, ``<out_dir>/labels/*.csv`` and ``manifest.csv``."""
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    if limit:
        sources = sources[:limit]
    jobs = []
    for src in sources:
        base = Path(src).stem
        for k in range(copies + 1):
            jobs.append(
                Job(
                    src=src,
                    out_audio=str(out_dir / "audio" / f"{base}_{k}.flac"),
                    out_csv=str(out_dir / "labels" / f"{base}_{k}.csv"),
                    copy=k,
                    seed=seed,
                    p_aug=p_aug,
                    stretch_method=stretch_method,
                )
            )
    results = []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    it = None
    if progress:
        from tqdm import tqdm

        it = tqdm(total=len(jobs), unit="file", desc="make-data")
    if workers == 1:
        for j in jobs:
            results.append(_run_job(j))
            if it:
                it.update(1)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_job, j) for j in jobs]
            for f in as_completed(futs):
                results.append(f.result())
                if it:
                    it.update(1)
    if it:
        it.close()
    results.sort(key=lambda r: (r["src"], r["copy"]))
    skipped = [r for r in results if "skipped" in r]
    if skipped:
        print(
            f"make-data: {len(skipped)} job(s) skipped:",
            *[f"  {r['src']} copy {r['copy']}: {r['skipped']}" for r in skipped[:10]],
            sep="\n",
        )
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio", "labels", "source", "copy", "n_chunks", "n_degraded", "counts"])
        for r in results:
            if "skipped" in r:
                continue
            w.writerow(
                [r["audio"], r["labels"], r["src"], r["copy"], r["n_chunks"], r["n_degraded"], json.dumps(r["counts"])]
            )
    with open(out_dir / "make_data_config.json", "w") as f:
        json.dump(
            {
                "copies": copies,
                "seed": seed,
                "p_aug": p_aug,
                "stretch_method": stretch_method,
                "augment_params": AUGMENT_PARAMS,
                "sample_rate": SAMPLE_RATE,
                "segment_seconds": SEGMENT_SECONDS,
                "n_sources": len(sources),
                "n_files": sum("skipped" not in r for r in results),
            },
            f,
            indent=2,
        )
    return results
