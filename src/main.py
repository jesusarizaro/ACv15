#!/usr/bin/env python3
import argparse
from app_platform import ensure_dirs, CFG_DIR, REP_DIR
from configio import load_config, save_config

def cmd_setup():
    ensure_dirs()
    cfg = load_config()
    save_config(cfg)
    print(f"Config lista en {CFG_DIR/'config.yaml'}")

def cmd_doctor():
    ensure_dirs()
    issues=[]
    cfg = load_config()
    from pathlib import Path
    if not Path(cfg["reference"]["wav_path"]).exists():
        issues.append("No existe el archivo de referencia (config.reference.wav_path).")
    print("✅ Dirs OK:", str(REP_DIR.parent))
    if issues:
        print("⚠️  Observaciones:")
        for s in issues: print("  -", s)
        return 1
    print("✅ Config OK.")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--doctor", action="store_true")
    args = ap.parse_args()
    if args.setup: cmd_setup()
    elif args.doctor: exit(cmd_doctor())
    else: ap.print_help()

if __name__ == "__main__":
    main()
