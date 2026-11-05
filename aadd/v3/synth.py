"""v3 degradation synthesiser: event-based, frame-labelled, co-occurring degradations.

One call to :meth:`Synthesizer.make` turns a clean speech window into a degraded
window plus a ``[n_frames, n_classes]`` target in {0, 1} and an ignore mask.

Signal chain (fixed order, every event is applied to the whole signal and then
crossfaded in only over its interval)::

    pitch_shift, time_stretch -> reverb -> noise, music, hum, clicks
      -> lowpass, highpass -> clipping -> codec -> packet_loss

Frame labels are 1 over the event interval, restricted for some classes to the
frames where the effect is physically observable (see ``_observable``); frames
within ``boundary_ignore`` of an interval edge and reverb tails are masked out.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

SR = 16000
FRAME = 320  # 20 ms
CLASSES_V3 = [
    "noise",
    "music",
    "reverb",
    "clipping",
    "lowpass",
    "highpass",
    "codec",
    "packet_loss",
    "hum",
    "clicks",
    "pitch_shift",
    "time_stretch",
]
CI = {c: i for i, c in enumerate(CLASSES_V3)}
ORDER = [
    "pitch_shift",
    "time_stretch",
    "reverb",
    "noise",
    "music",
    "hum",
    "clicks",
    "lowpass",
    "highpass",
    "clipping",
    "codec",
    "packet_loss",
]
NARROWBAND_CODECS = ["GSM610", "ULAW", "ALAW", "G721_32", "IMA_ADPCM"]


@dataclass
class Event:
    cls: str
    start: int  # samples
    end: int
    params: dict = field(default_factory=dict)


@dataclass
class Assets:
    """Pools the synthesiser draws from. Arrays are float32 mono 16 kHz."""

    noises: list  # list[np.ndarray] (or memmap slices)
    music: list
    rirs: list
    rir_meta: list | None = None
    babble: list | None = None  # speech pool for synthetic crowd noise (labelled "noise")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def _active_rms(x: np.ndarray, frame: int = 400) -> float:
    """RMS over the louder half of 25 ms frames (≈ active speech)."""
    n = len(x) // frame
    if n < 2:
        return _rms(x)
    e = (x[: n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(1)
    thr = np.median(e)
    sel = e[e >= thr]
    return float(np.sqrt(sel.mean() + 1e-12))


def _fit(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop or loop-pad ``x`` to length ``n``."""
    if len(x) >= n:
        o = int(rng.integers(0, len(x) - n + 1))
        return np.asarray(x[o : o + n], dtype=np.float32)
    reps = math.ceil(n / max(len(x), 1))
    return np.tile(np.asarray(x, dtype=np.float32), reps)[:n]


def _crossfade_mask(n: int, start: int, end: int, fade: int = 160) -> np.ndarray:
    m = np.zeros(n, dtype=np.float32)
    m[start:end] = 1.0
    f = min(fade, (end - start) // 2)
    if f > 0:
        if start > 0:
            m[start : start + f] = np.linspace(0, 1, f, dtype=np.float32)
        if end < n:
            m[end - f : end] = np.linspace(1, 0, f, dtype=np.float32)
    return m


class Synthesizer:
    def __init__(
        self,
        assets: Assets,
        seconds: float = 8.0,
        p_events: tuple = (0.2, 0.35, 0.3, 0.15),
        p_whole: float = 0.4,
        min_dur: float = 0.5,
        max_dur: float = 4.0,
        boundary_ignore: float = 0.04,
        class_weights: dict | None = None,
    ):
        self.a = assets
        self.n = int(seconds * SR)
        self.n_frames = self.n // FRAME
        self.p_events = np.asarray(p_events) / np.sum(p_events)
        self.p_whole = p_whole
        self.min_dur, self.max_dur = int(min_dur * SR), int(max_dur * SR)
        self.bi = round(boundary_ignore * SR / FRAME)
        w = np.array([(class_weights or {}).get(c, 1.0) for c in CLASSES_V3], dtype=np.float64)
        self.class_p = w / w.sum()

    # ------------------------------------------------------------------ events
    def sample_events(self, rng: np.random.Generator) -> list[Event]:
        k = int(rng.choice(len(self.p_events), p=self.p_events))
        classes = list(rng.choice(CLASSES_V3, size=k, replace=False, p=self.class_p)) if k else []
        evs = []
        for c in classes:
            if rng.random() < self.p_whole and c not in ("clicks", "packet_loss"):
                s, e = 0, self.n
            else:
                d = int(rng.integers(self.min_dur, self.max_dur + 1))
                s = int(rng.integers(0, self.n - d + 1))
                e = s + d
            evs.append(Event(str(c), s, e))
        return evs

    # ------------------------------------------------------------------ main
    def make(self, speech: np.ndarray, rng: np.random.Generator, events: list[Event] | None = None):
        """Return (audio float32 [n], target float32 [F, C], ignore bool [F, C], events)."""
        x = _fit(speech, self.n, rng)
        peak = np.abs(x).max() + 1e-9
        x = x * (float(rng.uniform(0.2, 0.9)) / peak)  # random level, no pre-clipping
        clean = x.copy()
        events = self.sample_events(rng) if events is None else events
        F, C = self.n_frames, len(CLASSES_V3)
        target = np.zeros((F, C), dtype=np.float32)
        ignore = np.zeros((F, C), dtype=bool)
        speech_rms = _active_rms(clean)
        # frames of near-silence in the clean source (edits there are inaudible)
        fe = (clean[: F * FRAME].reshape(F, FRAME) ** 2).mean(1)
        silent = fe < (speech_rms**2) * 10 ** (-40 / 10)

        for cls in ORDER:
            for ev in [e for e in events if e.cls == cls]:
                y, obs = getattr(self, f"_{cls}")(x, ev, rng, speech_rms)
                m = _crossfade_mask(self.n, ev.start, ev.end)
                x = (x * (1 - m) + y * m).astype(np.float32)
                self._label(target, ignore, ev, obs, silent)
                if cls == "codec" and ev.params.get("subtype") in NARROWBAND_CODECS:
                    self._label(target, ignore, Event("lowpass", ev.start, ev.end), None, silent)
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
        return x, target, ignore, events

    def _label(self, target, ignore, ev: Event, obs: np.ndarray | None, silent: np.ndarray):
        c = CI[ev.cls]
        f0, f1 = ev.start // FRAME, math.ceil(ev.end / FRAME)
        f1 = min(f1, self.n_frames)
        lab = np.zeros(self.n_frames, dtype=bool)
        lab[f0:f1] = True
        if obs is not None:  # restrict to observable frames, ignore the rest of the interval
            ign = lab & ~obs
            lab = lab & obs
            ignore[ign, c] = True
        if ev.cls in ("pitch_shift", "time_stretch"):
            ignore[lab & silent, c] = True
            lab = lab & ~silent
        target[lab, c] = 1.0
        # boundary uncertainty
        for b in (f0, f1):
            if 0 < b < self.n_frames:
                lo, hi = max(0, b - self.bi), min(self.n_frames, b + self.bi)
                ignore[lo:hi, c] = True
        if ev.cls == "reverb" and f1 < self.n_frames:  # reverberant tail after the interval
            ignore[f1 : min(self.n_frames, f1 + 15), c] = True

    def _frames_changed(self, x, y, rel_db: float = -40.0, ref: float | None = None) -> np.ndarray:
        F = self.n_frames
        d = ((y - x)[: F * FRAME].reshape(F, FRAME) ** 2).mean(1)
        ref = ref if ref is not None else _active_rms(x) ** 2
        return d > ref * 10 ** (rel_db / 10)

    # ------------------------------------------------------------------ transforms
    # each returns (processed full signal, observable-frame mask or None)

    def _babble(self, rng) -> np.ndarray:
        k = int(rng.integers(4, 11))
        out = np.zeros(self.n, dtype=np.float32)
        for _ in range(k):
            sp = _fit(self.a.babble[int(rng.integers(len(self.a.babble)))], self.n, rng)
            out += sp / (_rms(sp) + 1e-9) * float(rng.uniform(0.5, 1.0))
        return out

    def _noise(self, x, ev, rng, srms):
        u = rng.random()
        if self.a.babble is not None and 0.12 <= u < 0.32:  # 20 %: crowd of 4-10 other talkers
            n = self._babble(rng)
            ev.params["source"] = "babble"
        elif u < 0.12 or not len(self.a.noises):  # 12 %: white / pink / brown noise
            colour = str(rng.choice(["white", "pink", "brown"]))
            n = rng.standard_normal(self.n).astype(np.float32)
            if colour != "white":
                f = np.fft.rfftfreq(self.n, 1 / SR)
                f[0] = f[1]
                spec = np.fft.rfft(n) / (np.sqrt(f) if colour == "pink" else f)
                n = np.fft.irfft(spec, n=self.n).astype(np.float32)
            ev.params["source"] = f"colour:{colour}"
        else:  # real recorded noise (MUSAN, FSD50K)
            i = int(rng.integers(len(self.a.noises)))
            n = _fit(self.a.noises[i], self.n, rng)
            ev.params["source"] = f"pool:{i}"
        snr = float(rng.uniform(-5, 30))
        ev.params["snr_db"] = snr
        g = srms / (_rms(n) + 1e-9) * 10 ** (-snr / 20)
        return x + g * n, None

    def _music(self, x, ev, rng, srms):
        if not self.a.music:
            return self._noise(x, ev, rng, srms)
        i = int(rng.integers(len(self.a.music)))
        m = _fit(self.a.music[i], self.n, rng)
        snr = float(rng.uniform(0, 25))
        ev.params.update(source=f"musan-music:{i}", snr_db=snr)
        return x + srms / (_rms(m) + 1e-9) * 10 ** (-snr / 20) * m, None

    def _reverb(self, x, ev, rng, srms):
        i = int(rng.integers(len(self.a.rirs)))
        h = np.asarray(self.a.rirs[i], dtype=np.float32)
        k = int(np.argmax(np.abs(h)))
        h = h[max(0, k - 16) :]  # align direct path
        h = h[: SR * 2]
        h = h / (np.abs(h).max() + 1e-9)
        y = sps.fftconvolve(x, h)[: self.n].astype(np.float32)
        y *= _rms(x) / (_rms(y) + 1e-9)
        ev.params["rir"] = i
        if self.a.rir_meta:
            ev.params.update(self.a.rir_meta[i])
        return y, None

    def _hum(self, x, ev, rng, srms):
        f0 = float(rng.choice([50.0, 60.0])) * float(rng.uniform(0.995, 1.005))
        nh = int(rng.integers(3, 11))
        a = float(rng.uniform(0.5, 1.5))
        t = np.arange(self.n) / SR
        jitter = 1 + 0.002 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t)
        h = sum(
            (1 / k**a) * np.sin(2 * np.pi * f0 * k * t * jitter + rng.uniform(0, 2 * np.pi)) for k in range(1, nh + 1)
        )
        lvl = float(rng.uniform(-35, -15))
        ev.params.update(f0=f0, harmonics=nh, level_db=lvl)
        return x + (srms / (_rms(h) + 1e-9) * 10 ** (lvl / 20) * h).astype(np.float32), None

    def _clicks(self, x, ev, rng, srms):
        y = x.copy()
        rate = float(rng.uniform(2, 20))
        dur = (ev.end - ev.start) / SR
        k = max(1, int(rng.poisson(rate * dur)))
        pos = rng.integers(ev.start, max(ev.start + 1, ev.end - 50), size=k)
        peak = np.abs(x).max() + 1e-9
        obs = np.zeros(self.n_frames, dtype=bool)
        for p in pos:
            L = int(rng.integers(2, 48))  # 0.1-3 ms
            amp = float(rng.uniform(0.1, 0.8)) * peak * rng.choice([-1, 1])
            pulse = amp * np.hanning(L + 2)[1:-1] * rng.uniform(0.5, 1.0, L)
            if rng.random() < 0.5:  # resonant click
                b, a = sps.iirpeak(float(rng.uniform(1000, 6000)), 5.0, SR)
                pulse = sps.lfilter(b, a, np.pad(pulse, (0, 64)))
            e = min(self.n, p + len(pulse))
            y[p:e] += pulse[: e - p].astype(np.float32)
            obs[p // FRAME : min(self.n_frames, e // FRAME + 1)] = True
        ev.params.update(rate=rate, n_clicks=int(k))
        return y, obs

    def _lowpass(self, x, ev, rng, srms):
        if rng.random() < 0.4:
            import soxr

            sr2 = int(rng.choice([4000, 6000, 8000]))
            y = soxr.resample(soxr.resample(x, SR, sr2), sr2, SR)[: self.n]
            y = np.pad(y, (0, self.n - len(y))).astype(np.float32)
            ev.params.update(kind="resample", sr=sr2)
            return y, None
        fc = float(rng.uniform(1000, 5000))
        order = int(rng.integers(4, 9))
        sos = sps.butter(order, fc, "low", fs=SR, output="sos")
        ev.params.update(kind="butter", cutoff=fc, order=order)
        return sps.sosfilt(sos, x).astype(np.float32), None

    def _highpass(self, x, ev, rng, srms):
        fc = float(rng.uniform(200, 1000))
        order = int(rng.integers(2, 9))
        sos = sps.butter(order, fc, "high", fs=SR, output="sos")
        ev.params.update(cutoff=fc, order=order)
        return sps.sosfilt(sos, x).astype(np.float32), None

    def _clipping(self, x, ev, rng, srms):
        seg = np.abs(x[ev.start : ev.end])
        if rng.random() < 0.6:
            frac = float(rng.uniform(0.02, 0.30))
            thr = float(np.quantile(seg, 1 - frac)) if len(seg) else 1.0
            thr = max(thr, 1e-4)
            y = np.clip(x, -thr, thr)
            ev.params.update(kind="hard", clipped_fraction=frac)
        else:
            drive = float(rng.uniform(2, 10))
            pk = seg.max() + 1e-9 if len(seg) else 1.0
            y = (np.tanh(drive * x / pk) / np.tanh(drive) * pk).astype(np.float32)
            ev.params.update(kind="tanh", drive=drive)
        obs = self._frames_changed(x, y, rel_db=-35.0)
        return y.astype(np.float32), obs

    def _codec(self, x, ev, rng, srms):
        import soundfile as sf

        kind = rng.choice(["MP3", "OPUS", "VORBIS", "NB"], p=[0.35, 0.25, 0.1, 0.3])
        buf = io.BytesIO()
        if kind == "NB":
            sub = str(rng.choice(NARROWBAND_CODECS))
            import soxr

            z = soxr.resample(x, SR, 8000).astype(np.float32)
            fmt = "WAV"
            sf.write(buf, np.clip(z, -1, 1), 8000, format=fmt, subtype=sub)
            buf.seek(0)
            d, _ = sf.read(buf, dtype="float32")
            y = soxr.resample(d, 8000, SR).astype(np.float32)
            ev.params.update(codec=sub)
            ev.params["subtype"] = sub
        else:
            fmt, sub = ("MP3", "MPEG_LAYER_III") if kind == "MP3" else ("OGG", kind)
            # levels chosen so the bitrate is audibly low: MP3 CBR 8-41 kbps, Opus 10-32 kbps, Vorbis ~30 kbps
            lo = {"MP3": 0.8, "OPUS": 0.9, "VORBIS": 0.8}[str(kind)]
            lvl = float(rng.uniform(lo, 0.99))
            kw = {"bitrate_mode": "CONSTANT"} if kind == "MP3" else {}
            sf.write(buf, np.clip(x, -1, 1), SR, format=fmt, subtype=sub, compression_level=lvl, **kw)
            buf.seek(0)
            y, _ = sf.read(buf, dtype="float32")
            ev.params.update(
                codec=kind, compression_level=lvl, kbps=round(8 * buf.getbuffer().nbytes / (len(x) / SR) / 1000, 1)
            )
        y = self._align(x, y)
        return y, None

    def _align(self, x, y, max_lag: int = 3000):
        """Remove codec delay by cross-correlation on a 1-s probe; crop/pad to n."""
        n = self.n
        probe = slice(SR, 2 * SR) if n >= 3 * SR else slice(0, min(n, SR))
        a = x[probe]
        best, lag = -1, 0
        if len(y) > probe.stop + max_lag:
            c = sps.correlate(y[probe.start : probe.stop + max_lag], a, mode="valid", method="fft")
            lag = int(np.argmax(c))
            best = c[lag]
        y = y[lag:] if best > 0 else y
        return np.pad(y[:n], (0, max(0, n - len(y)))).astype(np.float32)

    def _packet_loss(self, x, ev, rng, srms):
        pkt = int(rng.choice([160, 320, 640]))  # 10/20/40 ms
        loss = float(rng.uniform(0.05, 0.30))
        burst = float(rng.uniform(1, 5))
        p_bg = 1.0 / burst  # leave bad state
        p_gb = loss * p_bg / max(1e-6, 1 - loss)  # enter bad state
        y = x.copy()
        bad = False
        lost = np.zeros(self.n, dtype=bool)
        fill = rng.choice(["zero", "repeat"])
        for s in range(ev.start, ev.end - pkt + 1, pkt):
            bad = (rng.random() > p_bg) if bad else (rng.random() < p_gb)
            if bad:
                lost[s : s + pkt] = True
                y[s : s + pkt] = 0.0 if fill == "zero" or s - pkt < 0 else y[s - pkt : s]
        ev.params.update(packet_ms=pkt / 16, loss=loss, burst=burst, fill=str(fill))
        F = self.n_frames
        lf = lost[: F * FRAME].reshape(F, FRAME).any(1)
        obs = lf & self._frames_changed(x, y, rel_db=-30.0)
        return y, obs

    def _pitch_shift(self, x, ev, rng, srms):
        import librosa

        st = float(rng.uniform(2, 8)) * float(rng.choice([-1, 1]))
        s, e = max(0, ev.start - 1600), min(self.n, ev.end + 1600)
        y = x.copy()
        y[s:e] = librosa.effects.pitch_shift(x[s:e], sr=SR, n_steps=st, n_fft=1024)
        ev.params["semitones"] = st
        return y.astype(np.float32), None

    def _time_stretch(self, x, ev, rng, srms):
        import librosa

        rate = float(rng.uniform(0.7, 0.9)) if rng.random() < 0.5 else float(rng.uniform(1.15, 1.5))
        d = ev.end - ev.start
        # take a source span of length d*rate so that the stretched span is exactly d
        src_len = int(d * rate)
        s0 = max(0, min(ev.start, self.n - src_len))
        span = x[s0 : s0 + src_len]
        z = librosa.effects.time_stretch(span, rate=rate, n_fft=1024)
        z = np.pad(z[:d], (0, max(0, d - len(z))))
        y = x.copy()
        y[ev.start : ev.end] = z
        ev.params["rate"] = rate
        return y.astype(np.float32), None
