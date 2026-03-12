"""
Lie Detector Pro — Dashboard
Fixed: auto-start camera, black screen, import paths
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

from modules.webcam_analyzer import WebcamAnalyzer
from modules.voice_analyzer   import VoiceAnalyzer
from modules.fusion_engine    import FusionEngine, score_label

BG      = "#0a0e17"
PANEL   = "#111827"
BORDER  = "#1e293b"
ACCENT  = "#06b6d4"
TEXT    = "#e2e8f0"
SUBTEXT = "#64748b"
GREEN   = "#22c55e"
YELLOW  = "#f59e0b"
RED     = "#ef4444"

FONT_TITLE = ("Courier New", 20, "bold")
FONT_MONO  = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_BIG   = ("Courier New", 34, "bold")

CALIBRATION_SECS = 6

QUESTIONS = [
    "What is your full name?",
    "Are you currently employed?",
    "Have you ever stolen anything?",
    "Do you exercise regularly?",
    "Have you lied to someone today?",
    "Do you enjoy your current job?",
    "Have you ever cheated on a test?",
    "Are you happy right now?",
]


class LieDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LIE DETECTOR PRO")
        self.root.configure(bg=BG)
        self.root.geometry("1280x820")
        self.root.resizable(False, False)

        self.webcam = WebcamAnalyzer()
        self.voice  = VoiceAnalyzer()
        self.fusion = FusionEngine()

        self.phase     = "IDLE"
        self.cal_start = None
        self.running   = False
        self.score     = 0.0
        self._photo    = None

        self._build_ui()
        self._auto_start()
        self.root.after(50, self._tick)

    def _auto_start(self):
        opened = self.webcam.open_camera(0)
        if not opened:
            self.webcam.open_camera(1)
        try:
            self.voice.start()
        except Exception as e:
            print(f"Mic warning: {e}")
        self.running = True
        self.status_lbl.config(text="● LIVE — click CALIBRATE", fg=YELLOW)

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=BG, pady=6)
        hdr.pack(fill="x", padx=16)
        tk.Label(hdr, text="LIE DETECTOR PRO", font=FONT_TITLE,
                 fg=ACCENT, bg=BG).pack(side="left")
        self.status_lbl = tk.Label(hdr, text="STARTING...",
                                   font=FONT_MONO, fg=SUBTEXT, bg=BG)
        self.status_lbl.pack(side="right")

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=6)

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both")

        self.cam_label = tk.Label(left, bg="#000000")
        self.cam_label.pack()

        wf = tk.Frame(left, bg=PANEL)
        wf.pack(fill="x", pady=(4, 0))
        tk.Label(wf, text="VOICE WAVEFORM", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", padx=6, pady=2)
        self.wave_fig = Figure(figsize=(4.8, 1.1), facecolor=PANEL)
        self.wave_ax  = self.wave_fig.add_subplot(111)
        self.wave_ax.set_facecolor(PANEL)
        self.wave_ax.axis("off")
        self.wave_line, = self.wave_ax.plot([], [], color=ACCENT, lw=1)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=wf)
        self.wave_canvas.get_tk_widget().pack(fill="x")

        right = tk.Frame(body, bg=BG, padx=8)
        right.pack(side="left", fill="both", expand=True)

        sp = tk.Frame(right, bg=PANEL)
        sp.pack(fill="x", pady=(0, 5))
        self.score_lbl = tk.Label(sp, text="--.-", font=FONT_BIG, fg=GREEN, bg=PANEL)
        self.score_lbl.pack(pady=(8, 0))
        self.verdict_lbl = tk.Label(sp, text="CLICK CALIBRATE TO BEGIN",
                                    font=("Courier New", 12, "bold"), fg=YELLOW, bg=PANEL)
        self.verdict_lbl.pack(pady=(0, 4))
        self.bar_canvas = tk.Canvas(sp, height=12, bg=BORDER, highlightthickness=0)
        self.bar_canvas.pack(fill="x", padx=12, pady=(0, 8))

        mf = tk.Frame(right, bg=PANEL)
        mf.pack(fill="x", pady=(0, 5))
        tk.Label(mf, text="LIVE METRICS", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).grid(row=0, column=0, columnspan=4,
                                             sticky="w", padx=8, pady=(4, 2))
        self._mvars = {}
        items = [("BLINKS/MIN","blinks"),("EYE SHIFT","eye"),
                 ("HEAD MOVE","head"),  ("PITCH Hz","pitch"),
                 ("TREMOR","tremor"),   ("PAUSES","pause"),
                 ("VISUAL","vis"),      ("VOICE","voc")]
        for i, (title, key) in enumerate(items):
            r, c = divmod(i, 4)
            tk.Label(mf, text=title, font=FONT_SMALL,
                     fg=SUBTEXT, bg=PANEL).grid(row=r*2+1, column=c, padx=10, pady=(3,0))
            v = tk.StringVar(value="—")
            self._mvars[key] = v
            tk.Label(mf, textvariable=v, font=FONT_MONO,
                     fg=ACCENT, bg=PANEL).grid(row=r*2+2, column=c, padx=10, pady=(0,4))

        cf = tk.Frame(right, bg=PANEL)
        cf.pack(fill="both", expand=True, pady=(0, 5))
        tk.Label(cf, text="SCORE TIMELINE (30s)", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", padx=8, pady=2)
        self.chart_fig = Figure(figsize=(5.5, 1.9), facecolor=PANEL)
        self.chart_ax  = self.chart_fig.add_subplot(111)
        self.chart_ax.set_facecolor("#0d1520")
        self.chart_ax.set_ylim(0, 100)
        self.chart_ax.tick_params(colors=SUBTEXT, labelsize=7)
        for sp2 in self.chart_ax.spines.values():
            sp2.set_edgecolor(BORDER)
        self.chart_line, = self.chart_ax.plot([], [], color=ACCENT, lw=1.5)
        self.chart_fig.tight_layout(pad=0.4)
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=cf)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

        ctrl = tk.Frame(right, bg=BG)
        ctrl.pack(fill="x", pady=(0, 4))
        self.q_var = tk.StringVar(value=QUESTIONS[0])
        q_menu = tk.OptionMenu(ctrl, self.q_var, *QUESTIONS)
        q_menu.config(bg=PANEL, fg=TEXT, font=FONT_SMALL,
                      activebackground=BORDER, relief="flat", width=38)
        q_menu.pack(side="left", padx=(0, 4))
        for txt, cmd, bg, fg in [
            ("CALIBRATE", self._start_calibration, "#1e3a5f", ACCENT),
            ("ASK",       self._ask_question,      "#1a3a1a", GREEN),
            ("END",       self._end_question,       "#3a1a1a", RED),
            ("SAVE",      self._save_report,        PANEL,     TEXT),
        ]:
            tk.Button(ctrl, text=txt, command=cmd, font=FONT_SMALL,
                      bg=bg, fg=fg, relief="flat", padx=6).pack(side="left", padx=2)

        rf = tk.Frame(right, bg=PANEL)
        rf.pack(fill="x")
        tk.Label(rf, text="QUESTION RESULTS", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", padx=8, pady=(4, 2))
        self.results_text = tk.Text(rf, height=4, bg="#0d1520", fg=TEXT,
                                    font=FONT_SMALL, relief="flat", state="disabled")
        self.results_text.pack(fill="x", padx=6, pady=(0, 6))

    def _start_calibration(self):
        self.phase     = "CALIBRATING"
        self.cal_start = time.time()
        self.status_lbl.config(text="● CALIBRATING", fg=YELLOW)
        self.verdict_lbl.config(text="CALIBRATING — speak normally...", fg=YELLOW)

    def _ask_question(self):
        if self.phase not in ("READY", "DETECTING"):
            messagebox.showwarning("Not Ready", "Please calibrate first!")
            return
        self.fusion.start_question(self.q_var.get())
        self.phase = "DETECTING"
        self.status_lbl.config(text="● DETECTING", fg=RED)

    def _end_question(self):
        if self.phase != "DETECTING":
            return
        self.fusion.end_question()
        self.phase = "READY"
        self.status_lbl.config(text="● READY", fg=GREEN)
        self._refresh_results()

    def _save_report(self):
        path = self.fusion.save_report()
        messagebox.showinfo("Saved", f"Report saved:\n{path}")

    def _refresh_results(self):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        for r in reversed(self.fusion.question_scores[-6:]):
            self.results_text.insert(
                "end", f"[{r['label']:10s} {r['avg_score']:5.1f}]  {r['question']}\n")
        self.results_text.config(state="disabled")

    def _tick(self):
        try:
            self._update()
        except Exception as e:
            print(f"[tick error] {e}")
        self.root.after(40, self._tick)

    def _update(self):
        now = time.time()

        frame = self.webcam.process_frame()
        if frame is not None:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((480, 360), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.cam_label.config(image=photo)
            self._photo = photo

        if self.phase == "CALIBRATING":
            elapsed = now - self.cal_start
            pct = int(min(elapsed / CALIBRATION_SECS * 100, 100))
            self.verdict_lbl.config(
                text=f"CALIBRATING {pct}% — speak naturally", fg=YELLOW)
            if elapsed >= CALIBRATION_SECS:
                self.webcam.calibrate()
                self.voice.calibrate()
                self.phase = "READY"
                self.status_lbl.config(text="● READY", fg=GREEN)
                self.verdict_lbl.config(text="READY — select a question", fg=GREEN)
            return

        if self.phase in ("READY", "DETECTING"):
            vs  = self.webcam.get_visual_score()
            acs = self.voice.get_voice_score()
            self.score = self.fusion.update(vs, acs)
            label, color = score_label(self.score)

            self.score_lbl.config(text=f"{self.score:.1f}", fg=color)
            self.verdict_lbl.config(text=label, fg=color)

            w = self.bar_canvas.winfo_width()
            if w > 1:
                bw = int(w * self.score / 100)
                self.bar_canvas.delete("all")
                self.bar_canvas.create_rectangle(0, 0, bw, 12, fill=color, outline="")

            ws  = self.webcam.snapshot()
            vss = self.voice.snapshot()
            self._mvars["blinks"].set(f"{ws['blinks_per_min']:.1f}")
            self._mvars["eye"].set(f"{ws['eye_std']:.3f}")
            self._mvars["head"].set(f"{ws['head_avg']:.1f}px")
            self._mvars["pitch"].set(f"{vss['pitch_mean']:.0f}")
            self._mvars["tremor"].set(f"{vss['tremor']:.4f}")
            self._mvars["pause"].set(f"{vss['pause_ratio']:.2f}")
            self._mvars["vis"].set(f"{ws['visual_score']:.1f}")
            self._mvars["voc"].set(f"{vss['voice_score']:.1f}")

            times, scores = self.fusion.get_recent_scores(30)
            if len(scores) > 1:
                self.chart_line.set_data(times, scores)
                self.chart_ax.set_xlim(times[0], 0)
                self.chart_canvas.draw_idle()

        wf = self.voice.waveform
        if len(wf) > 1:
            x = np.linspace(0, 1, len(wf))
            self.wave_ax.set_xlim(0, 1)
            self.wave_ax.set_ylim(-0.3, 0.3)
            self.wave_line.set_data(x, wf)
            self.wave_canvas.draw_idle()

        self.voice.transcribe_async()

    def on_close(self):
        self.running = False
        self.webcam.release()
        self.voice.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = LieDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
