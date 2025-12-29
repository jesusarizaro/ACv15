#!/usr/bin/env python3
from app_platform import ensure_dirs, CFG_DIR, REP_DIR
from configio import load_config
from pathlib import Path

def main():
    ensure_dirs()
    cfg = load_config()
    print(f"✅ Config encontrada: {CFG_DIR/'config.yaml'}")
    REP_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Directorios de datos creados.")
    if not Path(cfg["reference"]["wav_path"]).exists():
        print("⚠️  Falta el archivo de referencia definido en config.reference.wav_path")
    else:
        print("✅ Archivo de referencia OK.")

if __name__ == "__main__":
    main()
