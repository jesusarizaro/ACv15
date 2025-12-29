#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import sounddevice as sd



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
    U = (win**2).sum()

    X = np.fft.rfft(frames * win[None, :], axis=1)
    Pxx = (np.abs(X)**2) / (fs * U)
    Pxx = Pxx.mean(axis=0)

    f = np.fft.rfftfreq(nperseg, 1/fs)
    Pxx_db = 10*np.log10(np.maximum(Pxx, 1e-30))
    return f.astype(np.float32), Pxx_db.astype(np.float32)

# =========================================================
# DETECCIÓN DE BANDERAS (FRECUENCIA)
# =========================================================

def detect_flag_offsets_by_freq(
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

    offsets = []

    for i in range(0, len(x) - win, hop):
        frame = x[i:i+win] * np.hanning(win)
        spec = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(win, 1/fs)
        mag_db = 20*np.log10(np.abs(spec) + 1e-12)

        mask = (freqs >= target_freq - freq_tol) & \
               (freqs <= target_freq + freq_tol)

        if np.any(mask) and np.max(mag_db[mask]) > threshold_db:
            if not offsets or i - offsets[-1] > min_sep:
                offsets.append(i)

    return offsets

# =========================================================
# RECORTE ENTRE PRIMERA Y ÚLTIMA BANDERA
# =========================================================

def crop_between_flags(
    x: np.ndarray,
    fs: int,
    flag_freq: float = 5500.0
) -> Tuple[np.ndarray, dict]:

    offsets = detect_flag_offsets_by_freq(x, fs, target_freq=flag_freq)

    if len(offsets) < 2:
        return x.copy(), {
            "cropped": False,
            "start_sample": None,
            "end_sample": None,
            "start_s": None,
            "end_s": None,
        }

    start = offsets[0]
    end = offsets[-1]

    return x[start:end], {
        "cropped": True,
        "start_sample": int(start),
        "end_sample": int(end),
        "start_s": round(start/fs, 4),
        "end_s": round(end/fs, 4),
    }

# =========================================================
# ANÁLISIS PRINCIPAL
# =========================================================

BANDS = {
    "LFE": (30, 100),
    "LF":  (30, 120),
    "MF":  (120, 2000),
    "HF":  (2000, 8000),
}

def band_energy_db(f, psd_db, band):
    f1, f2 = band
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return -120.0
    p = 10**(psd_db[mask]/10)
    return 10*np.log10(np.mean(p) + 1e-30)

#---------------------------------------------------------------------------------------------------------
def analyze_pair(x_ref: np.ndarray, x_cur: np.ndarray, fs: int) -> dict:
    rms_ref = rms_db(x_ref)
    rms_cur = rms_db(x_cur)

    crest_ref = crest_factor_db(x_ref)
    crest_cur = crest_factor_db(x_cur)

    peak_cur = float(np.max(np.abs(x_cur)) + 1e-12)

    # ===== DETECCIÓN DE CANAL MUERTO (CRÍTICO) =====
    dead_abs = (rms_cur < DEAD_RMS_DB) or (peak_cur < DEAD_PEAK)

    # PSD
    f_ref, psd_ref = welch_db(x_ref, fs)
    f_cur, psd_cur = welch_db(x_cur, fs)

    bands_ref = {k: band_energy_db(f_ref, psd_ref, v) for k, v in BANDS.items()}
    bands_cur = {k: band_energy_db(f_cur, psd_cur, v) for k, v in BANDS.items()}
    diff_bands = {k: bands_cur[k] - bands_ref[k] for k in BANDS}

    band_fail = any(abs(v) > 6 for v in diff_bands.values())
    crest_fail = abs(crest_cur - crest_ref) > 4

    overall = "FAILED" if (dead_abs or band_fail or crest_fail) else "PASSED"

    return {
        "Evaluacion": overall,
        "Estado": "MUERTO" if dead_abs else "VIVO",
        "rms": {
            "ref_db": rms_ref,
            "cin_db": rms_cur
        },
        "crest": {
            "ref_db": crest_ref,
            "cin_db": crest_cur
        },
        "spec95_db": 0.0,  # si luego lo usas
        "ref": bands_ref,
        "cine": bands_cur,
        "delta": diff_bands,
        "peak_cur": peak_cur,
        "dead_channel": bool(dead_abs)
    }


# =========================================================
# JSON
# =========================================================
def build_json_payload(
    fs: int,
    global_result: dict | None,
    channel_results: list[dict],
    ref_markers,
    cur_markers,
    ref_segments,
    cur_segments,
    ref_wav,
    cin_wav
) -> dict:
    """
    Construye payload MULTICANAL para ThingsBoard
    Keys: Canal1 ... Canal6
    """

    payload = {}

    for idx, ch in enumerate(channel_results):
        canal_id = f"Canal{idx + 1}"

        estado = "MUERTO" if ch.get("dead_channel") else "VIVO"

        payload[canal_id] = {
            "Evaluacion": ch.get("overall", "FAILED"),
            "Estado": estado,
            "ref": {
                k: float(v) for k, v in ch.get("bands_ref", {}).items()
            },
            "cine": {
                k: float(v) for k, v in ch.get("bands_cur", {}).items()
            },
            "delta": {
                k: float(v) for k, v in ch.get("diff_bands", {}).items()
            }
        }

    return payload





def json_safe(obj):
    """
    Convierte recursivamente tipos NumPy a tipos nativos de Python
    para que json.dump no falle.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):   # 🔴 ESTA ES LA CLAVE
        return bool(obj)
    else:
        return obj



# ===========================================================================================================NUEVO
# RECORTE POR BANDERAS DE FRECUENCIA (ACv5)
# ======================================================

def detect_frequency_flags(
    x: np.ndarray,
    fs: int,
    target_freq: float = 5500.0,
    freq_tol: float = 40.0,
    threshold_db: float = -40.0,
    win_ms: float = 30.0,
    hop_ms: float = 10.0,
    min_sep_s: float = 0.3
) -> list[int]:
    """
    Detecta múltiples banderas de frecuencia (beeps).
    Retorna índices de inicio (samples).
    """

    win = int(fs * win_ms * 1e-3)
    hop = int(fs * hop_ms * 1e-3)

    offsets = []

    for i in range(0, len(x) - win, hop):
        frame = x[i:i + win] * np.hanning(win)
        spec = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(win, d=1 / fs)
        mag_db = 20 * np.log10(np.abs(spec) + 1e-12)

        mask = (freqs >= target_freq - freq_tol) & \
               (freqs <= target_freq + freq_tol)

        if np.any(mask) and np.max(mag_db[mask]) > threshold_db:
            offsets.append(i)

    # eliminar detecciones muy cercanas
    clean = []
    min_sep = int(min_sep_s * fs)
    for o in offsets:
        if not clean or o - clean[-1] > min_sep:
            clean.append(o)

    return clean


def crop_between_frequency_flags(
    x: np.ndarray,
    fs: int,
    **kwargs
):
    """
    Recorta la señal entre la primera y última bandera.
    Retorna: original, recortado, fs, start_idx, end_idx
    """

    offsets = detect_frequency_flags(x, fs, **kwargs)

    if len(offsets) < 2:
        return x, x.copy(), fs, 0, len(x)

    start = offsets[0]
    end   = offsets[-1]

    return x, x[start:end], fs, start, end
