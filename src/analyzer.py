#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from datetime import datetime
from typing import List, Tuple, Optional, Any

import numpy as np
import sounddevice as sd

# =========================================================
# UMBRALES PARA CANAL MUERTO
# =========================================================
DEAD_RMS_DB = -55.0     # silencio real (mic normal)
DEAD_PEAK   = 0.02      # pico mínimo (señal normalizada)

# =========================================================
# UTILIDADES BÁSICAS DE AUDIO
# =========================================================

def normalize_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    m = np.max(np.abs(x)) if x.size else 0.0
    return x / (m + 1e-12) if m > 1.0 else x

def record_audio(
    duration_sec: float,
    fs: int = 48000,
    channels: int = 1,
    device: Optional[int] = None
) -> np.ndarray:
    duration_sec = max(0.5, float(duration_sec))
    kwargs = dict(samplerate=fs, channels=channels, dtype="float32")
    if device is not None:
        kwargs["device"] = device
    rec = sd.rec(int(duration_sec * fs), **kwargs)
    sd.wait()
    return normalize_mono(rec.squeeze())

def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(x**2) + 1e-20) + 1e-20)

def crest_factor_db(x: np.ndarray) -> float:
    peak = np.max(np.abs(x)) + 1e-20
    rms = np.sqrt(np.mean(x**2) + 1e-20)
    return 20.0 * np.log10(peak / rms)

# =========================================================
# PSD TIPO WELCH (SIN SCIPY)
# =========================================================

def _frame_signal(x: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    step = nperseg - noverlap
    n = len(x)
    nwin = 1 + max(0, (n - nperseg) // step)
    out = np.zeros((nwin, nperseg), dtype=np.float32)
    for i in range(nwin):
        s = i * step
        seg = x[s:s+nperseg]
        out[i, :len(seg)] = seg
    return out

def welch_db(x: np.ndarray, fs: int, nperseg: int = 4096):
    noverlap = nperseg // 2
    frames = _frame_signal(x, nperseg, noverlap)
    win = np.hanning(nperseg).astype(np.float32)
    U = (win**2).sum() + 1e-30

    X = np.fft.rfft(frames * win[None, :], axis=1)
    Pxx = (np.abs(X)**2) / (fs * U)
    Pxx = Pxx.mean(axis=0)

    f = np.fft.rfftfreq(nperseg, 1/fs)
    Pxx_db = 10*np.log10(np.maximum(Pxx, 1e-30))
    return f.astype(np.float32), Pxx_db.astype(np.float32)

# =========================================================
# DETECCIÓN DE BANDERAS + RECORTES
# =========================================================

def detect_frequency_flags(
    x: np.ndarray,
    fs: int,
    target_freq: float = 5500.0,
    freq_tol: float = 40.0,
    threshold_db: float = -40.0,
    win_ms: float = 30.0,
    hop_ms: float = 10.0,
    min_sep_s: float = 0.3
) -> List[int]:
    win = int(fs * win_ms * 1e-3)
    hop = int(fs * hop_ms * 1e-3)
    min_sep = int(min_sep_s * fs)

    offsets: List[int] = []
    if win <= 0 or hop <= 0 or len(x) < win:
        return offsets

    for i in range(0, len(x) - win, hop):
        frame = x[i:i + win] * np.hanning(win)
        spec = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(win, d=1 / fs)
        mag_db = 20 * np.log10(np.abs(spec) + 1e-12)

        mask = (freqs >= target_freq - freq_tol) & (freqs <= target_freq + freq_tol)
        if np.any(mask) and (float(np.max(mag_db[mask])) > float(threshold_db)):
            if not offsets or (i - offsets[-1] > min_sep):
                offsets.append(i)

    return offsets

def crop_between_frequency_flags(
    x: np.ndarray,
    fs: int,
    target_freq: float = 5500.0,
    **kwargs
):
    offsets = detect_frequency_flags(x, fs, target_freq=target_freq, **kwargs)
    if len(offsets) < 2:
        return x, x.copy(), fs, 0, len(x)

    start = int(offsets[0])
    end   = int(offsets[-1])
    end = max(start + 1, min(end, len(x)))
    return x, x[start:end], fs, start, end

# =========================================================
# SPLIT EN CANALES USANDO BANDERAS INTERNAS (4.5 kHz)
# =========================================================

def split_by_markers(x: np.ndarray, markers: List[int], n_channels: int = 6) -> List[np.ndarray]:
    """
    Construye n_channels segmentos usando boundaries = [0] + markers + [len(x)].
    Si no hay suficientes boundaries, cae a split_equal_segments.
    """
    markers = sorted(set(int(m) for m in markers if 0 <= int(m) <= len(x)))
    bounds = [0] + markers + [len(x)]

    # limpiar bounds muy cercanas (evita segmentos vacíos)
    clean = [bounds[0]]
    for b in bounds[1:]:
        if b - clean[-1] >= 10:  # mínimo 10 samples
            clean.append(b)
    bounds = clean

    if len(bounds) < (n_channels + 1):
        return split_equal_segments(x, n_channels)

    # usa las primeras n_channels+1 boundaries
    bounds = bounds[:(n_channels + 1)]
    segs = []
    for i in range(n_channels):
        a, b = bounds[i], bounds[i+1]
        segs.append(x[a:b])
    return segs

def split_equal_segments(x: np.ndarray, n: int = 6) -> List[np.ndarray]:
    if n <= 0:
        return [x]
    L = len(x)
    if L == 0:
        return [x.copy() for _ in range(n)]
    size = max(1, L // n)
    segs = []
    for i in range(n):
        a = i * size
        b = (i + 1) * size if i < n - 1 else L
        segs.append(x[a:b])
    return segs

def split_channels_by_internal_beeps(
    x: np.ndarray,
    fs: int,
    n_channels: int = 6,
    marker_freq: float = 4500.0,
    freq_tol: float = 40.0,
    threshold_db: float = -40.0
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Detecta beeps internos (4.5 kHz) y segmenta.
    Devuelve: (segments, markers)
    """
    markers = detect_frequency_flags(
        x, fs,
        target_freq=marker_freq,
        freq_tol=freq_tol,
        threshold_db=threshold_db
    )
    segs = split_by_markers(x, markers, n_channels=n_channels)
    return segs, markers

# =========================================================
# BANDAS PARA COMPARACIÓN
# =========================================================

BANDS = {
    "LFE": (30, 100),
    "LF":  (100, 250),
    "MF":  (250, 2000),
    "HF":  (2000, 8000),
}

def band_energy_db(f, psd_db, band):
    f1, f2 = band
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return -120.0
    p = 10**(psd_db[mask]/10)
    return 10*np.log10(np.mean(p) + 1e-30)

# =========================================================
# ANÁLISIS POR CANAL
# =========================================================

def analyze_pair(x_ref: np.ndarray, x_cur: np.ndarray, fs: int) -> dict:
    rms_ref = float(rms_db(x_ref))
    rms_cur = float(rms_db(x_cur))

    crest_ref = float(crest_factor_db(x_ref))
    crest_cur = float(crest_factor_db(x_cur))

    peak_cur = float(np.max(np.abs(x_cur)) + 1e-12)

    dead_abs = (rms_cur < DEAD_RMS_DB) or (peak_cur < DEAD_PEAK)

    f_ref, psd_ref = welch_db(x_ref, fs)
    f_cur, psd_cur = welch_db(x_cur, fs)

    ref = {k: float(band_energy_db(f_ref, psd_ref, v)) for k, v in BANDS.items()}
    cine = {k: float(band_energy_db(f_cur, psd_cur, v)) for k, v in BANDS.items()}
    delta = {k: float(cine[k] - ref[k]) for k in BANDS}

    band_fail = any(abs(v) > 6.0 for v in delta.values())
    crest_fail = abs(crest_cur - crest_ref) > 4.0

    evaluacion = "FAILED" if (dead_abs or band_fail or crest_fail) else "PASSED"

    return {
        "Evaluacion": evaluacion,
        "Estado": "MUERTO" if dead_abs else "VIVO",
        "ref": ref,
        "cine": cine,
        "delta": delta,
        "rms": {"ref_db": rms_ref, "cin_db": rms_cur},
        "crest": {"ref_db": crest_ref, "cin_db": crest_cur},
        "peak_cur": peak_cur,
        "dead_channel": bool(dead_abs),
    }

# =========================================================
# PAYLOAD THINGSBOARD (Canal1..Canal6)
# =========================================================

def build_json_payload(
    fs: int,
    global_result: Optional[dict],
    channel_results: List[dict],
    *args, **kwargs
) -> dict:
    payload: dict = {}
    for i, ch in enumerate(channel_results, start=1):
        payload[f"Canal{i}"] = {
            "Evaluacion": ch.get("Evaluacion", "FAILED"),
            "Estado": ch.get("Estado", "MUERTO"),
            "ref": ch.get("ref", {}),
            "cine": ch.get("cine", {}),
            "delta": ch.get("delta", {}),
        }
    return payload

# =========================================================
# JSON SAFE
# =========================================================

def json_safe(obj: Any):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
