#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import yaml
from app_platform import CFG_DIR, ASSETS_DIR

CFG_PATH = CFG_DIR / "config.yaml"

DEFAULTS = {
    "general": {"oncalendar": "*-*-* 02:00:00"},
    "audio": {
        "fs": 48000,
        "duration_s": 10.0,
        "prefer_input_name": "",
    },
    "thingsboard": {
        "host": "thingsboard.cloud",
        "port": 1883,
        "use_tls": False,
        "token": "",
    },
    "reference": {
        "wav_path": str((ASSETS_DIR / "reference_master.wav").resolve())
    },
    "noise": {
        "wav_path": str((ASSETS_DIR / "noise_floor.wav").resolve())
    },
}

def _ensure_dirs():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def load_config() -> dict:
    _ensure_dirs()
    data = {}
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    # merge superficial + asegurar secciones
    out = DEFAULTS | data
    for k in ("general","audio","thingsboard","reference","noise"):
        out.setdefault(k, {})
        out[k] = DEFAULTS[k] | out[k]
    if "file" in out["reference"] and "wav_path" not in out["reference"]:
        out["reference"]["wav_path"] = out["reference"]["file"]
    return out

def save_config(cfg: dict) -> None:
    _ensure_dirs()
    for k in ("general","audio","thingsboard","reference","noise"):
        cfg.setdefault(k, {})
        cfg[k] = DEFAULTS[k] | cfg[k]
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
