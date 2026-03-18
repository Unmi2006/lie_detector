"""
Lie Detector Pro — ULTIMATE+ EDITION
New: ML Classifier | Excel Export | Dark/Light Theme | Fullscreen | Session History
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time, threading

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

from modules.webcam_analyzer  import WebcamAnalyzer
from modules.voice_analyzer   import VoiceAnalyzer
from modules.fusion_engine    import FusionEngine, score_label
from modules.emotion_analyzer import EmotionAnalyzer
from modules.rppg_analyzer    import RPPGAnalyzer
from modules.language_pack    import get_lang, available_languages, DEFAULT_LANG
from modules.pdf_report       import generate_pdf_report
from modules.ml_classifier    import MLClassifier
from modules.session_history  import SessionHistoryWindow
from modules.excel_exporter   import export_excel, export_csv

# ─── Themes ──────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "BG": "#0a0e17", "PANEL": "#111827", "BORDER": "#1e293b",
        "ACCENT": "#06b6d4", "TEXT": "#e2e8f0", "SUBTEXT": "#64748b",
        "GREEN": "#22c55e", "YELLOW": "#f59e0b", "RED": "#ef4444",
        "PINK": "#ec4899", "CHART_BG": "#0d1520",
    },
    "light": {
        "BG": "#f1f5f9", "PANEL": "#ffffff", "BORDER": "#cbd5e1",
        "ACCENT": "#0284c7", "TEXT": "#0f172a", "SUBTEXT": "#64748b",
        "GREEN": "#16a34a", "YELLOW": "#d97706", "RED": "#dc2626",
        "PINK": "#db2777", "CHART_BG": "#f8fafc",
    },
}

EMOTION_COLORS = {
    "happy":"#22c55e","neutral":"#64748b","sad":"#60a5fa",
    "surprise":"#f59e0b","fear":"#f97316","angry":"#ef4444","disgust":"#a855f7",
}
SCORE_COLORS = {
    "TRUTH":"#22c55e","UNCERTAIN":"#38bdf8","SUSPECT":"#f97316","DECEPTION":"#ef4444"
}

FONT_TITLE = ("Courier New", 18, "bold")
FONT_MONO  = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_BIG   = ("Courier New", 32, "bold")
CALIBRATION_SECS = 6
ALERT_THRESHOLD  = 72
ALERT_COOLDOWN   = 8


def _beep():
    def _p():
        if HAS_WINSOUND:
            try: winsound.Beep(880,200); time.sleep(0.1); winsound.Beep(1100,300)
            except: pass
    threading.Thread(target=_p, daemon=True).start()


class LieDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.resizable(True, True)

        # Modules
        self.webcam  = WebcamAnalyzer()
        self.voice   = VoiceAnalyzer()
        self.emotion = EmotionAnalyzer(interval=0.8)
        self.rppg    = RPPGAnalyzer()
        self.fusion  = FusionEngine()
        self.ml      = MLClassifier()

        # State
        self.phase       = "IDLE"
        self.cal_start   = None
        self.running     = False
        self.score       = 0.0
        self._photo      = None
        self._last_alert = 0.0
        self.sound_on    = tk.BooleanVar(value=True)
        self.fullscreen  = tk.BooleanVar(value=False)
        self.theme_name  = "dark"
        self.T           = THEMES["dark"]

        # Language
        self.lang_name = tk.StringVar(value=DEFAULT_LANG)
        self.L         = get_lang(DEFAULT_LANG)
        self.all_questions = list(self.L["questions"])

        # ML labelling
        self._pending_label = None   # set after end_question, before label applied

        self._build_ui()
        self._apply_theme()
        self._auto_start()
        self.root.after(50, self._tick)

    # ── Startup ───────────────────────────────────────────────────────────────
    def _auto_start(self):
        if not self.webcam.open_camera(0):
            self.webcam.open_camera(1)
        try: self.voice.start()
        except Exception as e: print(f"Mic: {e}")
        self.emotion.start()
        self.rppg.start()
        self.running = True
        self.status_lbl.config(text=self.L["status_live"], fg=self.T["YELLOW"])

    # ── UI Build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        T = self.T
        # Top bar
        top = tk.Frame(self.root, pady=5)
        top.pack(fill="x", padx=14)
        self.title_lbl = tk.Label(top, font=FONT_TITLE)
        self.title_lbl.pack(side="left")

        # Right controls in header
        hdr_right = tk.Frame(top)
        hdr_right.pack(side="right")

        # Fullscreen toggle
        tk.Checkbutton(hdr_right, text="⛶ FS", variable=self.fullscreen,
                       font=FONT_SMALL, command=self._toggle_fullscreen).pack(side="right", padx=4)

        # Theme toggle
        self.theme_btn = tk.Button(hdr_right, text="☀ LIGHT", font=FONT_SMALL,
                                   relief="flat", padx=6, command=self._toggle_theme)
        self.theme_btn.pack(side="right", padx=4)

        # History button
        tk.Button(hdr_right, text="📋 HISTORY", font=FONT_SMALL,
                  relief="flat", padx=6,
                  command=self._open_history).pack(side="right", padx=4)

        # Language
        lf = tk.Frame(hdr_right)
        lf.pack(side="right", padx=4)
        self.lang_menu = tk.OptionMenu(lf, self.lang_name, *available_languages(),
                                       command=self._on_lang_change)
        self.lang_menu.config(font=FONT_SMALL, relief="flat", width=14)
        self.lang_menu.pack()

        self.status_lbl = tk.Label(top, font=FONT_MONO)
        self.status_lbl.pack(side="right", padx=10)

        self.sep = tk.Frame(self.root, height=1)
        self.sep.pack(fill="x")

        # Body
        self.body = tk.Frame(self.root)
        self.body.pack(fill="both", expand=True, padx=8, pady=5)

        # LEFT
        self.left = tk.Frame(self.body)
        self.left.pack(side="left", fill="both")

        self.cam_label = tk.Label(self.left, width=460, height=330)
        self.cam_label.pack()

        # Heart rate
        self.hr_pnl = tk.Frame(self.left)
        self.hr_pnl.pack(fill="x", pady=(3,0))
        hr_hdr = tk.Frame(self.hr_pnl)
        hr_hdr.pack(fill="x", padx=8, pady=(4,1))
        self.hr_title = tk.Label(hr_hdr, text="HEART RATE (rPPG)", font=FONT_SMALL)
        self.hr_title.pack(side="left")
        self.bpm_lbl = tk.Label(hr_hdr, text="-- BPM",
                                font=("Courier New",11,"bold"))
        self.bpm_lbl.pack(side="right")
        self.pulse_fig = Figure(figsize=(4.6,0.7), facecolor="#111827")
        self.pulse_ax  = self.pulse_fig.add_subplot(111)
        self.pulse_ax.set_facecolor("#0d1520"); self.pulse_ax.axis("off")
        self.pulse_line, = self.pulse_ax.plot([], [], color="#ec4899", lw=1.2)
        self.pulse_canvas = FigureCanvasTkAgg(self.pulse_fig, master=self.hr_pnl)
        self.pulse_canvas.get_tk_widget().pack(fill="x")

        # Emotion bars
        self.emo_pnl = tk.Frame(self.left)
        self.emo_pnl.pack(fill="x", pady=(3,0))
        emo_hdr = tk.Frame(self.emo_pnl)
        emo_hdr.pack(fill="x", padx=8, pady=(4,1))
        self.emo_title = tk.Label(emo_hdr, text="EMOTION", font=FONT_SMALL)
        self.emo_title.pack(side="left")
        self.dominant_lbl = tk.Label(emo_hdr, font=("Courier New",10,"bold"))
        self.dominant_lbl.pack(side="right")
        self.emo_bars = {}; self.emo_lbls = {}
        for em in ["happy","neutral","surprise","fear","angry","disgust","sad"]:
            row = tk.Frame(self.emo_pnl)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=f"{em:8s}", font=FONT_SMALL, width=8, anchor="w").pack(side="left")
            bar = tk.Canvas(row, height=8, highlightthickness=0, width=240)
            bar.pack(side="left", padx=(2,4))
            lbl = tk.Label(row, text="0%", font=FONT_SMALL, width=5)
            lbl.pack(side="left")
            self.emo_bars[em] = bar; self.emo_lbls[em] = lbl
        tk.Frame(self.emo_pnl, height=2).pack()

        # Waveform
        self.wf_pnl = tk.Frame(self.left)
        self.wf_pnl.pack(fill="x", pady=(3,0))
        self.wf_title = tk.Label(self.wf_pnl, text="VOICE WAVEFORM", font=FONT_SMALL)
        self.wf_title.pack(anchor="w", padx=6, pady=2)
        self.wave_fig = Figure(figsize=(4.6,0.7), facecolor="#111827")
        self.wave_ax  = self.wave_fig.add_subplot(111)
        self.wave_ax.set_facecolor("#111827"); self.wave_ax.axis("off")
        self.wave_line, = self.wave_ax.plot([], [], color="#06b6d4", lw=1)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=self.wf_pnl)
        self.wave_canvas.get_tk_widget().pack(fill="x")

        # RIGHT
        self.right = tk.Frame(self.body, padx=8)
        self.right.pack(side="left", fill="both", expand=True)

        # Score
        self.sp = tk.Frame(self.right)
        self.sp.pack(fill="x", pady=(0,4))
        self.score_lbl = tk.Label(self.sp, text="--.-", font=FONT_BIG)
        self.score_lbl.pack(pady=(6,0))
        self.verdict_lbl = tk.Label(self.sp, font=("Courier New",12,"bold"))
        self.verdict_lbl.pack(pady=(0,3))
        self.bar_canvas = tk.Canvas(self.sp, height=10, highlightthickness=0)
        self.bar_canvas.pack(fill="x", padx=10, pady=(0,6))

        # ML panel
        self.ml_pnl = tk.Frame(self.right)
        self.ml_pnl.pack(fill="x", pady=(0,4))
        ml_hdr = tk.Frame(self.ml_pnl)
        ml_hdr.pack(fill="x", padx=8, pady=(4,2))
        tk.Label(ml_hdr, text="ML CLASSIFIER", font=FONT_SMALL).pack(side="left")
        self.ml_status = tk.Label(ml_hdr, text="NOT TRAINED", font=FONT_SMALL)
        self.ml_status.pack(side="right")

        ml_row = tk.Frame(self.ml_pnl)
        ml_row.pack(fill="x", padx=8, pady=(0,4))
        # Truth/Lie label buttons
        tk.Button(ml_row, text="✓ TRUTH", font=FONT_SMALL, relief="flat",
                  padx=5, command=lambda: self._label_question(0)).pack(side="left", padx=(0,4))
        tk.Button(ml_row, text="✗ LIE", font=FONT_SMALL, relief="flat",
                  padx=5, command=lambda: self._label_question(1)).pack(side="left", padx=(0,4))
        tk.Button(ml_row, text="⚙ TRAIN", font=FONT_SMALL, relief="flat",
                  padx=5, command=self._train_ml).pack(side="left", padx=(0,4))
        self.ml_prob_lbl = tk.Label(ml_row, text="ML: —", font=FONT_SMALL)
        self.ml_prob_lbl.pack(side="right")

        # Signal bars
        self.sig_pnl = tk.Frame(self.right)
        self.sig_pnl.pack(fill="x", pady=(0,4))
        tk.Label(self.sig_pnl, text="SIGNALS", font=FONT_SMALL).pack(anchor="w", padx=8, pady=(4,2))
        self.sig_bars = {}; self.sig_lbls2 = {}
        for name, col, key in [("VISUAL","#06b6d4","vs"),("VOICE","#a3e635","vc"),
                                ("EMOTION","#a855f7","em"),("HEART ♥","#ec4899","hr"),
                                ("ML PROB","#f59e0b","ml")]:
            row = tk.Frame(self.sig_pnl)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=f"{name:9s}", font=FONT_SMALL, width=9, anchor="w").pack(side="left")
            bar = tk.Canvas(row, height=8, highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True, padx=(2,4))
            lbl = tk.Label(row, text="0.0", font=FONT_SMALL, width=5)
            lbl.pack(side="left")
            self.sig_bars[key] = (bar, col); self.sig_lbls2[key] = lbl
        tk.Frame(self.sig_pnl, height=2).pack()

        # Metrics
        self.mf = tk.Frame(self.right)
        self.mf.pack(fill="x", pady=(0,4))
        tk.Label(self.mf, text="LIVE METRICS", font=FONT_SMALL).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(4,2))
        self._mvars = {}
        for i,(title,key) in enumerate([
            ("BLINKS","blinks"),("EYE","eye"),("HEAD","head"),("PITCH","pitch"),
            ("TREMOR","tremor"),("PAUSES","pause"),("BPM","bpm"),("FUSED","fused"),
        ]):
            r,c = divmod(i,4)
            tk.Label(self.mf, text=title, font=FONT_SMALL).grid(row=r*2+1,column=c,padx=8,pady=(2,0))
            v = tk.StringVar(value="—"); self._mvars[key]=v
            tk.Label(self.mf, textvariable=v, font=FONT_MONO).grid(row=r*2+2,column=c,padx=8,pady=(0,3))

        # Chart
        self.cf = tk.Frame(self.right)
        self.cf.pack(fill="both", expand=True, pady=(0,4))
        self.chart_title = tk.Label(self.cf, text="SCORE TIMELINE (30s)", font=FONT_SMALL)
        self.chart_title.pack(anchor="w", padx=8, pady=2)
        self.chart_fig = Figure(figsize=(5.0,1.5), facecolor="#111827")
        self.chart_ax  = self.chart_fig.add_subplot(111)
        self.chart_ax.set_facecolor("#0d1520"); self.chart_ax.set_ylim(0,100)
        self.chart_ax.tick_params(colors="#64748b", labelsize=7)
        for s in self.chart_ax.spines.values(): s.set_edgecolor("#1e293b")
        self.chart_line, = self.chart_ax.plot([],[],color="#06b6d4",lw=1.5)
        self.chart_fig.tight_layout(pad=0.3)
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=self.cf)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Controls
        ctrl = tk.Frame(self.right)
        ctrl.pack(fill="x", pady=(0,3))
        self.q_var = tk.StringVar(value=self.all_questions[0])
        self.q_menu = tk.OptionMenu(ctrl, self.q_var, *self.all_questions)
        self.q_menu.config(font=FONT_SMALL, relief="flat", width=28)
        self.q_menu.pack(side="left", padx=(0,3))
        self.add_q_btn = tk.Button(ctrl, text="+ ADD", font=FONT_SMALL,
                                   relief="flat", padx=4, command=self._add_question)
        self.add_q_btn.pack(side="left", padx=2)
        tk.Checkbutton(ctrl, text="🔔", variable=self.sound_on,
                       font=FONT_SMALL).pack(side="right")

        # Export buttons
        exp_row = tk.Frame(self.right)
        exp_row.pack(fill="x", pady=(0,3))
        self.btn_refs = {}
        for txt_key, cmd, col in [
            ("calibrate", self._start_cal,   "#1e3a5f"),
            ("ask",       self._ask,         "#1a3a1a"),
            ("end",       self._end,         "#3a1a1a"),
            ("save",      self._save_pdf,    "#0f1f2f"),
        ]:
            b = tk.Button(exp_row, text="", command=cmd, font=FONT_SMALL,
                          relief="flat", padx=5)
            b.pack(side="left", padx=2)
            self.btn_refs[txt_key] = b

        tk.Button(exp_row, text="📊 EXCEL", font=FONT_SMALL, relief="flat",
                  padx=5, command=self._save_excel).pack(side="left", padx=2)
        tk.Button(exp_row, text="📄 CSV", font=FONT_SMALL, relief="flat",
                  padx=5, command=self._save_csv).pack(side="left", padx=2)

        # Results
        self.rf = tk.Frame(self.right)
        self.rf.pack(fill="x")
        self.results_title = tk.Label(self.rf, text="QUESTION RESULTS", font=FONT_SMALL)
        self.results_title.pack(anchor="w", padx=8, pady=(3,1))
        self.results_text = tk.Text(self.rf, height=3, font=FONT_SMALL,
                                    relief="flat", state="disabled")
        self.results_text.pack(fill="x", padx=6, pady=(0,4))

        self._collect_themed_widgets()

    def _collect_themed_widgets(self):
        """Collect all widgets that need recoloring on theme change."""
        pass   # Done dynamically in _apply_theme via winfo_children recursion

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.T = THEMES[self.theme_name]
        self._apply_theme()
        self.theme_btn.config(text="☀ LIGHT" if self.theme_name=="dark" else "🌙 DARK")

    def _apply_theme(self):
        T = self.T
        def _recolor(widget):
            cls = widget.winfo_class()
            try:
                if cls in ("Frame","Toplevel","Tk"):
                    widget.config(bg=T["BG"])
                elif cls == "Label":
                    widget.config(bg=T["BG"] if widget.master.winfo_class() in ("Frame","Tk","Toplevel") else T["PANEL"],
                                  fg=T["TEXT"])
                elif cls in ("Button","Checkbutton"):
                    widget.config(bg=T["BG"], fg=T["TEXT"],
                                  activebackground=T["BORDER"],
                                  selectcolor=T["BG"])
                elif cls == "Text":
                    widget.config(bg=T["CHART_BG"], fg=T["TEXT"])
                elif cls == "Canvas":
                    widget.config(bg=T["BORDER"])
                elif cls == "Listbox":
                    widget.config(bg=T["PANEL"], fg=T["TEXT"])
            except Exception:
                pass
            for child in widget.winfo_children():
                _recolor(child)
        _recolor(self.root)

        # Fix specific panels
        for pnl in [self.sp, self.ml_pnl, self.sig_pnl, self.mf,
                    self.cf, self.rf, self.hr_pnl, self.emo_pnl, self.wf_pnl]:
            try: pnl.config(bg=T["PANEL"])
            except: pass

        self.root.config(bg=T["BG"])
        self.root.title(self.L["title"])
        self.title_lbl.config(fg=T["ACCENT"], bg=T["BG"])
        self.status_lbl.config(bg=T["BG"])
        self.sep.config(bg=T["BORDER"])
        self.score_lbl.config(bg=T["PANEL"])
        self.verdict_lbl.config(bg=T["PANEL"])
        self.bar_canvas.config(bg=T["BORDER"])
        self.chart_ax.set_facecolor(T["CHART_BG"])
        self.chart_fig.patch.set_facecolor(T["PANEL"])
        self.wave_ax.set_facecolor(T["PANEL"])
        self.wave_fig.patch.set_facecolor(T["PANEL"])
        self.chart_canvas.draw_idle()
        self.wave_canvas.draw_idle()

    # ── Fullscreen ────────────────────────────────────────────────────────────
    def _toggle_fullscreen(self):
        self.root.attributes("-fullscreen", self.fullscreen.get())

    # ── Language ──────────────────────────────────────────────────────────────
    def _on_lang_change(self, name):
        self.L = get_lang(name)
        self.all_questions = list(self.L["questions"])
        self._rebuild_q_menu()
        for key, btn in self.btn_refs.items():
            btn.config(text=self.L.get(key, key.upper()))
        self.root.title(self.L["title"])
        self.title_lbl.config(text=self.L["title"])

    def _rebuild_q_menu(self):
        menu = self.q_menu["menu"]
        menu.delete(0,"end")
        for q in self.all_questions:
            menu.add_command(label=q, command=lambda v=q: self.q_var.set(v))
        if self.all_questions: self.q_var.set(self.all_questions[0])

    def _add_question(self):
        q = simpledialog.askstring("Add Question","Type your question:", parent=self.root)
        if q and q.strip():
            self.all_questions.append(q.strip())
            self._rebuild_q_menu()
            self.q_var.set(q.strip())

    # ── Controls ──────────────────────────────────────────────────────────────
    def _start_cal(self):
        self.phase = "CALIBRATING"; self.cal_start = time.time()
        self.status_lbl.config(text=self.L["status_cal"], fg=self.T["YELLOW"])
        self.verdict_lbl.config(text=self.L["cal_prompt"].format(pct=0), fg=self.T["YELLOW"])

    def _ask(self):
        if self.phase not in ("READY","DETECTING"):
            messagebox.showwarning("!", self.L["not_ready"]); return
        q = self.q_var.get()
        self.fusion.start_question(q)
        self.ml.start_question()
        self.phase = "DETECTING"
        self.status_lbl.config(text=self.L["status_det"], fg=self.T["RED"])

    def _end(self):
        if self.phase != "DETECTING": return
        self.fusion.end_question()
        self.phase = "READY"
        self.status_lbl.config(text=self.L["status_ready"], fg=self.T["GREEN"])
        self._refresh_results()
        # Prompt for ML label
        self._prompt_ml_label()

    def _prompt_ml_label(self):
        win = tk.Toplevel(self.root)
        win.title("Label for ML Training")
        win.configure(bg=self.T["BG"])
        win.geometry("320x120")
        win.grab_set()
        tk.Label(win, text="Was the last answer a LIE or TRUTH?\n(Used to train your ML model)",
                 font=FONT_SMALL, bg=self.T["BG"], fg=self.T["TEXT"]).pack(pady=10)
        btn_row = tk.Frame(win, bg=self.T["BG"])
        btn_row.pack()
        tk.Button(btn_row, text="✓ TRUTH", font=FONT_SMALL, bg="#1a3a1a",
                  fg=self.T["GREEN"], relief="flat", padx=10,
                  command=lambda: [self._label_question(0), win.destroy()]).pack(side="left", padx=8)
        tk.Button(btn_row, text="✗ LIE", font=FONT_SMALL, bg="#3a1a1a",
                  fg=self.T["RED"], relief="flat", padx=10,
                  command=lambda: [self._label_question(1), win.destroy()]).pack(side="left", padx=8)
        tk.Button(btn_row, text="SKIP", font=FONT_SMALL, bg=self.T["PANEL"],
                  fg=self.T["SUBTEXT"], relief="flat", padx=6,
                  command=win.destroy).pack(side="left", padx=4)

    def _label_question(self, label):
        self.ml.label_last_question(label)
        snap = self.ml.snapshot()
        n = snap["n_samples"]
        needed = 20
        self.ml_status.config(
            text=f"TRAINED  {snap['accuracy']:.0f}%" if snap["trained"]
            else f"SAMPLES: {n}/{needed}")

    def _train_ml(self):
        result = self.ml.train()
        messagebox.showinfo("ML Training", result)
        snap = self.ml.snapshot()
        self.ml_status.config(
            text=f"TRAINED  {snap['accuracy']:.0f}%" if snap["trained"] else result)

    def _save_pdf(self):
        try:
            data = self.fusion.build_report_data()
            path = generate_pdf_report(data)
            messagebox.showinfo("✓", f"PDF saved:\n{path}")
        except Exception as e:
            messagebox.showerror("PDF Error", str(e))

    def _save_excel(self):
        try:
            data = self.fusion.build_report_data()
            path = export_excel(data)
            messagebox.showinfo("✓", f"Excel saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Excel Error", str(e))

    def _save_csv(self):
        try:
            data = self.fusion.build_report_data()
            path = export_csv(data)
            messagebox.showinfo("✓", f"CSV saved:\n{path}")
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))

    def _open_history(self):
        SessionHistoryWindow(self.root)

    def _refresh_results(self):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0","end")
        for r in reversed(self.fusion.question_scores[-5:]):
            self.results_text.insert("end",
                f"[{r['label']:10s} {r['avg_score']:5.1f}]  {r['question']}\n")
        self.results_text.config(state="disabled")

    def _maybe_alert(self, score):
        if (self.sound_on.get() and score >= ALERT_THRESHOLD
                and self.phase=="DETECTING"
                and time.time()-self._last_alert > ALERT_COOLDOWN):
            self._last_alert = time.time()
            _beep()

    # ── Tick ──────────────────────────────────────────────────────────────────
    def _tick(self):
        try: self._update()
        except Exception as e: print(f"[tick] {e}")
        self.root.after(40, self._tick)

    def _update(self):
        T = self.T; now = time.time()

        # Webcam
        frame = self.webcam.process_frame()
        if frame is not None:
            self.emotion.feed_frame(frame)
            lm = (self.webcam.face_mesh.process(
                cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            ).multi_face_landmarks or [None])[0]
            lm_list = lm.landmark if lm else None
            h,w = frame.shape[:2]
            self.rppg.feed_frame(frame, lm_list, w, h)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((460,330), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.cam_label.config(image=photo); self._photo = photo

        # rPPG
        rsnap = self.rppg.snapshot()
        bpm   = rsnap["bpm"]
        self.bpm_lbl.config(
            text=f"{bpm:.0f} BPM" if bpm>0 else "-- BPM",
            fg=T["PINK"] if rsnap["confident"] else T["SUBTEXT"])
        ps = self.rppg.pulse_signal
        if len(ps)>1:
            x=np.linspace(0,1,len(ps))
            self.pulse_ax.set_xlim(0,1); self.pulse_ax.set_ylim(-1.2,1.2)
            self.pulse_line.set_data(x,ps); self.pulse_canvas.draw_idle()

        # Emotions
        esnap    = self.emotion.snapshot()
        emotions = esnap.get("emotions",{})
        dominant = esnap.get("dominant","neutral")
        dom_conf = esnap.get("confidence",0.0)
        self.dominant_lbl.config(
            text=f"{dominant.upper()} {dom_conf:.0f}%",
            fg=EMOTION_COLORS.get(dominant,T["ACCENT"]))
        for em,bar in self.emo_bars.items():
            val=emotions.get(em,0.0); w2=bar.winfo_width()
            bar.delete("all")
            filled=int(w2*val/100)
            if filled>0:
                bar.create_rectangle(0,0,filled,8,
                    fill=EMOTION_COLORS.get(em,T["ACCENT"]),outline="")
            self.emo_lbls[em].config(text=f"{val:.0f}%")

        # Calibration
        if self.phase=="CALIBRATING":
            elapsed=now-self.cal_start; pct=int(min(elapsed/CALIBRATION_SECS*100,100))
            self.verdict_lbl.config(text=self.L["cal_prompt"].format(pct=pct),fg=T["YELLOW"])
            self.emotion.collect_calibration(); self.rppg.collect_calibration()
            if elapsed>=CALIBRATION_SECS:
                self.webcam.calibrate(); self.voice.calibrate()
                self.emotion.calibrate(); self.rppg.calibrate()
                self.phase="READY"
                self.status_lbl.config(text=self.L["status_ready"],fg=T["GREEN"])
                self.verdict_lbl.config(text=self.L["ready_prompt"],fg=T["GREEN"])
            return

        # Detection
        if self.phase in ("READY","DETECTING"):
            vs  = self.webcam.get_visual_score()
            acs = self.voice.get_voice_score()
            es  = self.emotion.get_emotion_lie_score()
            hrs = self.rppg.get_hr_lie_score()
            self.score = self.fusion.update(vs,acs,es,hrs)
            label,color = score_label(self.score)

            self.score_lbl.config(text=f"{self.score:.1f}",fg=color)
            self.verdict_lbl.config(text=label,fg=color)
            w2=self.bar_canvas.winfo_width()
            if w2>1:
                bw=int(w2*self.score/100)
                self.bar_canvas.delete("all")
                self.bar_canvas.create_rectangle(0,0,bw,10,fill=color,outline="")

            # Collect ML sample
            ws  = self.webcam.snapshot()
            vss = self.voice.snapshot()
            self.ml.collect_sample(
                vs,acs,es,hrs,
                ws["blinks_per_min"],ws["eye_std"],ws["head_avg"],
                vss["pitch_mean"],vss["tremor"],vss["pause_ratio"],
                rsnap["bpm"], esnap.get("dominant","neutral")
            )
            ml_prob = self.ml.predict_live({})

            # Signal bars
            for key,val in [("vs",vs),("vc",acs),("em",es),("hr",hrs),("ml",ml_prob)]:
                bar,col = self.sig_bars[key]; bw2=bar.winfo_width()
                bar.delete("all")
                filled=int(bw2*val/100)
                if filled>0:
                    bar.create_rectangle(0,0,filled,8,fill=col,outline="")
                self.sig_lbls2[key].config(text=f"{val:.1f}")

            # ML label
            if self.ml.trained:
                self.ml_prob_lbl.config(
                    text=f"ML: {ml_prob:.1f}%",
                    fg=T["RED"] if ml_prob>60 else T["GREEN"])

            # Metrics
            self._mvars["blinks"].set(f"{ws['blinks_per_min']:.1f}")
            self._mvars["eye"].set(f"{ws['eye_std']:.3f}")
            self._mvars["head"].set(f"{ws['head_avg']:.1f}")
            self._mvars["pitch"].set(f"{vss['pitch_mean']:.0f}")
            self._mvars["tremor"].set(f"{vss['tremor']:.4f}")
            self._mvars["pause"].set(f"{vss['pause_ratio']:.2f}")
            self._mvars["bpm"].set(f"{rsnap['bpm']:.0f}")
            self._mvars["fused"].set(f"{self.score:.1f}")

            # Timeline
            times,scores=self.fusion.get_recent_scores(30)
            if len(scores)>1:
                self.chart_line.set_data(times,scores)
                self.chart_ax.set_xlim(times[0],0)
                self.chart_canvas.draw_idle()

            self._maybe_alert(self.score)

        # Waveform
        wf=self.voice.waveform
        if len(wf)>1:
            x=np.linspace(0,1,len(wf))
            self.wave_ax.set_xlim(0,1); self.wave_ax.set_ylim(-0.3,0.3)
            self.wave_line.set_data(x,wf); self.wave_canvas.draw_idle()

        self.voice.transcribe_async()

    def on_close(self):
        self.running=False
        self.webcam.release(); self.voice.stop()
        self.emotion.stop(); self.rppg.stop()
        self.root.destroy()


if __name__=="__main__":
    root=tk.Tk()
    # Initial language apply
    app=LieDetectorApp(root)
    app._on_lang_change(DEFAULT_LANG)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
