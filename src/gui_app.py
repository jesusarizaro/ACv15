#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Any

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app_platform import APP_DIR, ASSETS_DIR, ensure_dirs
from configio import load_config, save_config
from iot_tb import send_json_to_thingsboard

from analyzer import (
    normalize_mono,
    record_audio,
    analyze_pair,
    build_json_payload,
    crop_between_frequency_flags,
    split_channels_by_internal_beeps,
    json_safe
)

APP_NAME = "AudioCinema"
SAVE_DIR = (APP_DIR / "data" / "captures").absolute()
EXPORT_DIR = (APP_DIR / "data" / "reports").absolute()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ENV_INPUT_INDEX = os.environ.get("AUDIOCINEMA_INPUT_INDEX")

INFO_TEXT = (
    "AudioCinema\n\n"
    "Esta aplicación graba, evalúa y compara una pista de PRUEBA con una "
    "pista de REFERENCIA para verificar el estado del sistema de audio.\n\n"
    "Qué hace:\n"
    "• Graba la pista de prueba con el micrófono.\n"
    "• Segmenta 6 canales (uno tras otro) usando beeps de 4.5kHz.\n"
    "• Compara cada canal vs referencia (bandas + RMS + crest + muerto).\n"
    "• Exporta JSON y (opcional) lo envía a ThingsBoard.\n"
)

# ---------- util mic ----------
def pick_input_device(preferred_name_substr: Optional[str] = None) -> Optional[int]:
    import sounddevice as sd
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    if ENV_INPUT_INDEX:
        try:
            idx = int(ENV_INPUT_INDEX)
            if 0 <= idx < len(devices) and devices[idx].get("max_input_channels", 0) > 0:
                return idx
        except Exception:
            pass

    if preferred_name_substr:
        s = preferred_name_substr.lower()
        for i, d in enumerate(devices):
            if s in str(d.get("name", "")).lower() and d.get("max_input_channels", 0) > 0:
                return i

    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            return i
    return None

# ---------- decorador para mostrar errores UI ----------
def ui_action(fn):
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                messagebox.showerror(APP_NAME, f"{e}\n\n{tb_str}")
            except Exception:
                print(tb_str)
            return None
    return wrapper


class AudioCinemaGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title(APP_NAME)

        tb.Style(theme="flatly")
        try:
            self.root.configure(bg="#e6e6e6")
        except Exception:
            pass

        self._icon_img = None
        try:
            icon_path = ASSETS_DIR / "audiocinema.png"
            if icon_path.exists():
                self._icon_img = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, self._icon_img)
        except Exception:
            self._icon_img = None

        ensure_dirs()
        self.cfg = load_config()

        self.fs = tk.IntVar(value=int(self._cfg(["audio","fs"], 48000)))
        self.duration = tk.DoubleVar(value=float(self._cfg(["audio","duration_s"], 60.0)))

        self.input_device_index: Optional[int] = None
        self.test_name = tk.StringVar(value="—")
        self.eval_text = tk.StringVar(value="—")

        self._build_ui()
        self._auto_select_input_device()

    # --- helpers cfg seguros ---
    def _cfg(self, path: List[str], default: Any = None) -> Any:
        d = self.cfg
        for key in path:
            if not isinstance(d, dict) or key not in d:
                return default
            d = d[key]
        return d

    def _set_cfg(self, path: List[str], value: Any) -> None:
        d = self.cfg
        for key in path[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[path[-1]] = value

    # --------------------- UI ---------------------
    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=8)
        root_frame.pack(fill=BOTH, expand=True)

        paned = ttk.Panedwindow(root_frame, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)

        # --------- IZQUIERDA ---------
        left = ttk.Frame(paned, padding=(6, 6))
        paned.add(left, weight=1)

        card = ttk.Frame(left, padding=6)
        card.pack(fill=Y, expand=False)

        if self._icon_img is not None:
            ttk.Label(card, image=self._icon_img).pack(anchor="n", pady=(0, 4))
        ttk.Label(card, text="AudioCinema", font=("Segoe UI", 18, "bold")).pack(anchor="n")

        desc = ("Graba, evalúa y analiza tu sistema de audio "
                "para garantizar la mejor experiencia envolvente.")
        ttk.Label(card, text=desc, wraplength=220, justify="center").pack(anchor="n", pady=(6, 10))

        btn_style = {"bootstyle": PRIMARY, "width": 20}
        tb.Button(card, text="Información",   command=self._show_info, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Configuración", command=self._popup_config, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Confirmación",  command=self._popup_confirm, **btn_style).pack(pady=6, fill=X)
        tb.Button(card, text="Prueba ahora",  command=self._run_once, **btn_style).pack(pady=(6, 0), fill=X)

        # separador vertical
        sep = ttk.Separator(root_frame, orient=VERTICAL)
        paned.add(sep)

        # --------- DERECHA ---------
        right = ttk.Frame(paned, padding=(8, 6))
        paned.add(right, weight=4)

        header = ttk.Frame(right)
        header.pack(fill=X, pady=(0, 8))

        ttk.Label(header, text="PRUEBA:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(header, textvariable=self.test_name, width=32, state="readonly", justify="center").grid(row=0, column=1, sticky="w")

        ttk.Label(header, text="EVALUACIÓN:", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(6, 0))
        self.eval_lbl = ttk.Label(header, textvariable=self.eval_text, font=("Segoe UI", 11, "bold"), foreground="#333")
        self.eval_lbl.grid(row=1, column=1, sticky="w", pady=(6, 0))

        fig_card = ttk.Frame(right, padding=4)
        fig_card.pack(fill=BOTH, expand=True)

        self.fig = Figure(figsize=(7, 8), dpi=100)
        self.ax_ref_orig = self.fig.add_subplot(2, 2, 1)
        self.ax_ref_cut  = self.fig.add_subplot(2, 2, 2)
        self.ax_cur_orig = self.fig.add_subplot(2, 2, 3)
        self.ax_cur_cut  = self.fig.add_subplot(2, 2, 4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_card)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self._clear_waves()
        self.fig.tight_layout()

        msg_card = ttk.Frame(right, padding=4)
        msg_card.pack(fill=BOTH, expand=False, pady=(6, 0))
        ttk.Label(msg_card, text="Mensajes", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.msg_text = tk.Text(msg_card, height=6, wrap="word")
        self.msg_text.pack(fill=BOTH, expand=True)
        self._set_messages(["Listo. Presiona «Prueba ahora» para iniciar."])

    def _clear_waves(self):
        axes = [
            (self.ax_ref_orig, "Referencia – ORIGINAL"),
            (self.ax_ref_cut,  "Referencia – RECORTADA"),
            (self.ax_cur_orig, "Prueba – ORIGINAL"),
            (self.ax_cur_cut,  "Prueba – RECORTADA"),
        ]
        for ax, title in axes:
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid(True, axis="x", ls=":")

    def _plot_wave(self, ax, x: np.ndarray, fs: int):
        n = len(x)
        t = np.arange(n, dtype=np.float32) / float(fs if fs else 1)
        ax.plot(t, x, linewidth=0.8)
        ax.set_xlim(0.0, float(t[-1]) if n else 1.0)

    def _set_eval(self, passed: Optional[bool]):
        if passed is None:
            self.eval_text.set("—")
            self.eval_lbl.configure(foreground="#333333")
        elif passed:
            self.eval_text.set("PASSED")
            self.eval_lbl.configure(foreground="#0d8a00")
        else:
            self.eval_text.set("FAILED")
            self.eval_lbl.configure(foreground="#cc0000")

    def _set_messages(self, lines: List[str]):
        self.msg_text.delete("1.0", tk.END)
        for ln in lines:
            self.msg_text.insert(tk.END, "• " + ln + "\n")
        self.msg_text.see(tk.END)

    def _auto_select_input_device(self):
        pref = str(self._cfg(["audio", "preferred_input_name"], ""))
        self.input_device_index = pick_input_device(pref)

    @ui_action
    def _show_info(self):
        messagebox.showinfo(APP_NAME, INFO_TEXT)

    @ui_action
    def _popup_confirm(self):
        tb_cfg = {
            "host": self._cfg(["thingsboard", "host"], "thingsboard.cloud"),
            "port": self._cfg(["thingsboard", "port"], 1883),
            "use_tls": self._cfg(["thingsboard", "use_tls"], False),
            "token": self._cfg(["thingsboard", "token"], ""),
        }
        txt = (
            f"Archivo de referencia:\n  {self._cfg(['reference','wav_path'], str(ASSETS_DIR/'reference_master.wav'))}\n\n"
            f"Audio:\n  fs={self._cfg(['audio','fs'],48000)}  duración={self._cfg(['audio','duration_s'],60.0)} s\n"
            f"ThingsBoard:\n  host={tb_cfg['host']}  port={tb_cfg['port']}  TLS={tb_cfg['use_tls']}\n"
        )
        messagebox.showinfo("Confirmación", txt)

    # =========================================================
    # CONFIG + GRABAR REFERENCIA (RESTaurado)
    # =========================================================
    @ui_action
    def _popup_config(self):
        w = tk.Toplevel(self.root)
        w.title("Configuración")

        frm = ttk.Frame(w, padding=10); frm.pack(fill=BOTH, expand=True)
        nb = ttk.Notebook(frm); nb.pack(fill=BOTH, expand=True)

        # -------- General --------
        g = ttk.Frame(nb); nb.add(g, text="General")
        ref_var = tk.StringVar(value=self._cfg(["reference","wav_path"], str(ASSETS_DIR/"reference_master.wav")))
        oncal_var = tk.StringVar(value=self._cfg(["oncalendar"], "*-*-* 02:00:00"))
        ttk.Label(g, text="Archivo de referencia (.wav):").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(g, textvariable=ref_var, width=52).grid(row=0, column=1, sticky="we", pady=(6,2))
        ttk.Label(g, text="OnCalendar (systemd):").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(g, textvariable=oncal_var, width=30).grid(row=1, column=1, sticky="w", pady=(6,2))

        # -------- Audio --------
        a = ttk.Frame(nb); nb.add(a, text="Audio")
        fs_var = tk.IntVar(value=int(self._cfg(["audio","fs"], 48000)))
        dur_var = tk.DoubleVar(value=float(self._cfg(["audio","duration_s"], 60.0)))
        pref_in = tk.StringVar(value=self._cfg(["audio","preferred_input_name"], ""))
        ttk.Label(a, text="Sample Rate (Hz):").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=fs_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(a, text="Duración (s):").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=dur_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(a, text="Preferir dispositivo:").grid(row=2, column=0, sticky="w", pady=(6,2))
        ttk.Entry(a, textvariable=pref_in, width=28).grid(row=2, column=1, sticky="w")

        # -------- ThingsBoard --------
        t = ttk.Frame(nb); nb.add(t, text="ThingsBoard")
        host_var = tk.StringVar(value=self._cfg(["thingsboard","host"], "thingsboard.cloud"))
        port_var = tk.IntVar(value=int(self._cfg(["thingsboard","port"], 1883)))
        tls_var  = tk.BooleanVar(value=bool(self._cfg(["thingsboard","use_tls"], False)))
        token_var = tk.StringVar(value=self._cfg(["thingsboard","token"], ""))
        ttk.Label(t, text="Host:").grid(row=0, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=host_var, width=24).grid(row=0, column=1, sticky="w")
        ttk.Label(t, text="Port:").grid(row=1, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=port_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(t, text="Usar TLS (8883)", variable=tls_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,2))
        ttk.Label(t, text="Token:").grid(row=3, column=0, sticky="w", pady=(6,2))
        ttk.Entry(t, textvariable=token_var, width=40).grid(row=3, column=1, sticky="w")

        # -------- Pista de referencia (GRABAR + 2 GRÁFICAS) --------
        r = ttk.Frame(nb)
        nb.add(r, text="Pista de referencia")

        r.grid_columnconfigure(0, weight=1)
        r.grid_columnconfigure(1, weight=1)

        fig_ref = Figure(figsize=(8, 3), dpi=100)
        ax_ref_orig = fig_ref.add_subplot(1, 2, 1)
        ax_ref_cut  = fig_ref.add_subplot(1, 2, 2)

        for ax, title in [(ax_ref_orig, "Referencia – ORIGINAL"), (ax_ref_cut, "Referencia – RECORTADA")]:
            ax.set_title(title)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid(True, axis="x", linestyle=":", linewidth=0.8)

        canvas_ref = FigureCanvasTkAgg(fig_ref, master=r)
        canvas_ref.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=(10, 8))

        ref_path_var = tk.StringVar(value=ref_var.get())
        ref_date_var = tk.StringVar(value="—")

        ttk.Label(r, text="Ruta:").grid(row=1, column=0, sticky="e", padx=(10,2))
        ttk.Label(r, textvariable=ref_path_var, wraplength=520, justify="left").grid(row=1, column=1, sticky="w", padx=(2,10))

        ttk.Label(r, text="Fecha:").grid(row=2, column=0, sticky="e", padx=(10,2))
        ttk.Label(r, textvariable=ref_date_var).grid(row=2, column=1, sticky="w", padx=(2,10))

        def _plot_ref(x: np.ndarray, fs0: int):
            x = normalize_mono(x)
            x_o, x_cut, fs_u, s, e = crop_between_frequency_flags(x, fs0, target_freq=5500.0)

            ax_ref_orig.clear()
            ax_ref_cut.clear()

            t1 = np.arange(len(x_o)) / fs_u
            t2 = np.arange(len(x_cut)) / fs_u

            ax_ref_orig.plot(t1, x_o, lw=0.7)
            ax_ref_cut.plot(t2, x_cut, lw=0.7)

            ax_ref_orig.axvline(s / fs_u, ls="--", color="green")
            ax_ref_orig.axvline(e / fs_u, ls="--", color="red")

            ax_ref_orig.set_title("Referencia – ORIGINAL")
            ax_ref_cut.set_title("Referencia – RECORTADA")

            for ax in (ax_ref_orig, ax_ref_cut):
                ax.set_xlabel("Tiempo (s)")
                ax.set_ylabel("Amplitud")
                ax.grid(True, axis="x", linestyle=":", linewidth=0.8)

            fig_ref.tight_layout()
            canvas_ref.draw_idle()

        def cargar_referencia_existente():
            try:
                ref_path = Path(ref_var.get()).resolve()
                if not ref_path.exists():
                    return
                x, fs0 = sf.read(ref_path, dtype="float32", always_2d=False)
                _plot_ref(x, fs0)
                ref_path_var.set(str(ref_path))
                mtime = datetime.fromtimestamp(ref_path.stat().st_mtime)
                ref_date_var.set(mtime.strftime("%Y-%m-%d %H:%M:%S"))
            except Exception as e:
                print("No se pudo cargar referencia:", e)

        def grabar_referencia():
            fs_now = int(fs_var.get())
            dur_now = float(dur_var.get())
            try:
                x = record_audio(dur_now, fs=fs_now, channels=1, device=self.input_device_index)
                ASSETS_DIR.mkdir(parents=True, exist_ok=True)
                out = (ASSETS_DIR / "reference_master.wav").resolve()
                sf.write(out, x, fs_now)

                ref_var.set(str(out))
                ref_path_var.set(str(out))
                ref_date_var.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                _plot_ref(x, fs_now)
                messagebox.showinfo(APP_NAME, f"Referencia guardada:\n{out}")
            except Exception as e:
                messagebox.showerror(APP_NAME, f"No se pudo grabar referencia:\n{e}")

        ttk.Button(r, text="Grabar Pista de Referencia", command=grabar_referencia).grid(
            row=3, column=0, columnspan=2, pady=(10, 14), sticky="n"
        )

        cargar_referencia_existente()

        # -------- Guardar / Cancelar --------
        btns = ttk.Frame(frm); btns.pack(fill=X, pady=(10,0))

        def on_save():
            self._set_cfg(["reference","wav_path"], ref_var.get().strip())
            self._set_cfg(["oncalendar"], oncal_var.get().strip())
            self._set_cfg(["audio","fs"], int(fs_var.get()))
            self._set_cfg(["audio","duration_s"], float(dur_var.get()))
            self._set_cfg(["audio","preferred_input_name"], pref_in.get().strip())
            self._set_cfg(["thingsboard","host"], host_var.get().strip())
            self._set_cfg(["thingsboard","port"], int(port_var.get()))
            self._set_cfg(["thingsboard","use_tls"], bool(tls_var.get()))
            self._set_cfg(["thingsboard","token"], token_var.get().strip())
            save_config(self.cfg)

            self.fs.set(int(self._cfg(["audio","fs"], 48000)))
            self.duration.set(float(self._cfg(["audio","duration_s"], 60.0)))

            messagebox.showinfo(APP_NAME, "Configuración guardada.")
            w.destroy()

        tb.Button(btns, text="Guardar", bootstyle=PRIMARY, command=on_save).pack(side=RIGHT)
        tb.Button(btns, text="Cancelar", bootstyle=SECONDARY, command=w.destroy).pack(side=RIGHT, padx=(0,6))

        try:
            w.grab_set()
            w.transient(self.root)
        except Exception:
            pass

    # =========================================================
    # RUN
    # =========================================================
    @ui_action
    def _run_once(self):
        fs  = int(self._cfg(["audio","fs"], 48000))
        dur = float(self._cfg(["audio","duration_s"], 60.0))

        ref_path = Path(self._cfg(["reference","wav_path"], str(ASSETS_DIR/"reference_master.wav")))
        if not ref_path.exists():
            raise FileNotFoundError(f"No existe archivo de referencia:\n{ref_path}")

        x_ref, fs_ref = sf.read(ref_path, dtype="float32", always_2d=False)
        if getattr(x_ref, "ndim", 1) == 2:
            x_ref = x_ref.mean(axis=1)
        x_ref = normalize_mono(x_ref)

        if fs_ref != fs:
            n_new = int(round(len(x_ref) * fs / fs_ref))
            x_idx = np.linspace(0, 1, len(x_ref))
            new_idx = np.linspace(0, 1, n_new)
            x_ref = np.interp(new_idx, x_idx, x_ref).astype(np.float32)

        x_cur = record_audio(dur, fs=fs, channels=1, device=self.input_device_index)

        # recorte global (5.5 kHz)
        x_ref_o, x_ref_cut, fs, ref_start, ref_end = crop_between_frequency_flags(x_ref, fs, target_freq=5500.0)
        x_cur_o, x_cur_cut, fs, cur_start, cur_end = crop_between_frequency_flags(x_cur, fs, target_freq=5500.0)

        # split 6 canales con beeps internos (4.5 kHz)
        ref_segs, ref_markers = split_channels_by_internal_beeps(x_ref_cut, fs, n_channels=6, marker_freq=4500.0)
        cur_segs, cur_markers = split_channels_by_internal_beeps(x_cur_cut, fs, n_channels=6, marker_freq=4500.0)

        channel_results = []
        for rseg, cseg in zip(ref_segs, cur_segs):
            channel_results.append(analyze_pair(rseg, cseg, fs))

        global_passed = all(ch["Evaluacion"] == "PASSED" for ch in channel_results)
        self._set_eval(global_passed)

        # gráficas
        self._clear_waves()

        self._plot_wave(self.ax_ref_orig, x_ref_o, fs)
        self.ax_ref_orig.axvline(ref_start / fs, color="green", ls="--")
        self.ax_ref_orig.axvline(ref_end   / fs, color="red",   ls="--")
        self._plot_wave(self.ax_ref_cut, x_ref_cut, fs)

        self._plot_wave(self.ax_cur_orig, x_cur_o, fs)
        self.ax_cur_orig.axvline(cur_start / fs, color="green", ls="--")
        self.ax_cur_orig.axvline(cur_end   / fs, color="red",   ls="--")
        self._plot_wave(self.ax_cur_cut, x_cur_cut, fs)

        self.canvas.draw_idle()

        self.test_name.set(datetime.now().strftime("Test_%Y-%m-%d_%H-%M-%S"))

        payload = build_json_payload(fs, None, channel_results)

        out = EXPORT_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)

        sent = False
        host = self._cfg(["thingsboard","host"], "thingsboard.cloud")
        port = int(self._cfg(["thingsboard","port"], 1883))
        token = self._cfg(["thingsboard","token"], "")
        use_tls = bool(self._cfg(["thingsboard","use_tls"], False))
        if token:
            sent = send_json_to_thingsboard(payload, host, port, token, use_tls)

        lines = []
        lines.append("La prueba ha " + ("aprobado." if global_passed else "fallado."))
        for i, ch in enumerate(channel_results, start=1):
            lines.append(f"Canal{i}: {ch['Estado']} / {ch['Evaluacion']}")
        lines.append(f"JSON: {out}")
        lines.append("Resultados enviados a ThingsBoard." if sent else "No se enviaron resultados a ThingsBoard.")
        self._set_messages(lines)

        messagebox.showinfo(APP_NAME, f"Análisis terminado.\nJSON: {out}")


# -------------------- main --------------------
def main():
    root = tb.Window(themename="flatly")
    app = AudioCinemaGUI(root)
    root.geometry("1020x640"); root.minsize(900,600)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
