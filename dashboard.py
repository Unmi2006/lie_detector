"""
Lie Detector Pro — ULTIMATE EDITION
Signals: Visual | Voice | Emotion (DeepFace) | Heart Rate (rPPG)
Extras:  Custom questions | Sound alert | Multi-language
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
import time
import threading
import winsound  # Windows; falls back gracefully on other OS

from modules.webcam_analyzer  import WebcamAnalyzer
from modules.voice_analyzer   import VoiceAnalyzer
from modules.fusion_engine    import FusionEngine, score_label
from modules.emotion_analyzer import EmotionAnalyzer
from modules.rppg_analyzer    import RPPGAnalyzer
from modules.language_pack    import get_lang, available_languages, DEFAULT_LANG

# ─── Palette ─────────────────────────────────────────────────────────────────
BG      = "#0a0e17"
PANEL   = "#111827"
BORDER  = "#1e293b"
ACCENT  = "#06b6d4"
TEXT    = "#e2e8f0"
SUBTEXT = "#64748b"
GREEN   = "#22c55e"
YELLOW  = "#f59e0b"
RED     = "#ef4444"
PURPLE  = "#a855f7"
PINK    = "#ec4899"

FONT_TITLE = ("Courier New", 18, "bold")
FONT_MONO  = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_BIG   = ("Courier New", 32, "bold")

CALIBRATION_SECS  = 6
ALERT_THRESHOLD   = 72   # score above this triggers sound alert
ALERT_COOLDOWN    = 8    # seconds between repeated alerts

EMOTION_COLORS = {
    "happy":   "#22c55e", "neutral": "#64748b", "sad":     "#60a5fa",
    "surprise":"#f59e0b", "fear":    "#f97316", "angry":   "#ef4444",
    "disgust": "#a855f7",
}


def _beep_alert():
    """Non-blocking alert sound — Windows beep, silent on other OS."""
    def _play():
        try:
            winsound.Beep(880, 200)
            time.sleep(0.1)
            winsound.Beep(1100, 300)
        except Exception:
            pass   # Non-Windows: skip silently
    threading.Thread(target=_play, daemon=True).start()


class LieDetectorApp:
    def __init__(self, root):
        self.root  = root
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # Modules
        self.webcam  = WebcamAnalyzer()
        self.voice   = VoiceAnalyzer()
        self.emotion = EmotionAnalyzer(interval=0.8)
        self.rppg    = RPPGAnalyzer()
        self.fusion  = FusionEngine()

        # State
        self.phase      = "IDLE"
        self.cal_start  = None
        self.running    = False
        self.score      = 0.0
        self._photo     = None
        self._last_alert = 0.0
        self.sound_on   = tk.BooleanVar(value=True)

        # Language
        self.lang_name = tk.StringVar(value=DEFAULT_LANG)
        self.L         = get_lang(DEFAULT_LANG)

        # Custom questions list
        self.all_questions = list(self.L["questions"])

        self._build_ui()
        self._apply_language()
        self._auto_start()
        self.root.after(50, self._tick)

    # ──────────────────────────────────────────────────────────────────────────
    def _auto_start(self):
        opened = self.webcam.open_camera(0)
        if not opened:
            self.webcam.open_camera(1)
        try:
            self.voice.start()
        except Exception as e:
            print(f"Mic warning: {e}")
        self.emotion.start()
        self.rppg.start()
        self.running = True
        self.status_lbl.config(text=self.L["status_live"], fg=YELLOW)

    # ──────────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG, pady=5)
        top.pack(fill="x", padx=14)

        self.title_lbl = tk.Label(top, text="", font=FONT_TITLE, fg=ACCENT, bg=BG)
        self.title_lbl.pack(side="left")

        # Language selector (right side of header)
        lang_frame = tk.Frame(top, bg=BG)
        lang_frame.pack(side="right")
        tk.Label(lang_frame, text="🌐", font=FONT_SMALL, fg=SUBTEXT, bg=BG).pack(side="left")
        lang_menu = tk.OptionMenu(lang_frame, self.lang_name,
                                  *available_languages(),
                                  command=self._on_language_change)
        lang_menu.config(bg=PANEL, fg=ACCENT, font=FONT_SMALL,
                         activebackground=BORDER, relief="flat", width=16)
        lang_menu.pack(side="left", padx=4)

        self.status_lbl = tk.Label(top, text="STARTING...",
                                   font=FONT_MONO, fg=SUBTEXT, bg=BG)
        self.status_lbl.pack(side="right", padx=10)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # ── Body ─────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=5)

        # LEFT ────────────────────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both")

        self.cam_label = tk.Label(left, bg="#000", width=480, height=340)
        self.cam_label.pack()

        # Heart rate panel
        hr_pnl = tk.Frame(left, bg=PANEL)
        hr_pnl.pack(fill="x", pady=(4, 0))
        hr_hdr = tk.Frame(hr_pnl, bg=PANEL)
        hr_hdr.pack(fill="x", padx=8, pady=(5,2))
        self.hr_title_lbl = tk.Label(hr_hdr, text="HEART RATE (rPPG)",
                                     font=FONT_SMALL, fg=SUBTEXT, bg=PANEL)
        self.hr_title_lbl.pack(side="left")
        self.bpm_lbl = tk.Label(hr_hdr, text="-- BPM",
                                font=("Courier New", 12, "bold"), fg=PINK, bg=PANEL)
        self.bpm_lbl.pack(side="right")
        # Pulse waveform
        self.pulse_fig = Figure(figsize=(4.8, 0.75), facecolor=PANEL)
        self.pulse_ax  = self.pulse_fig.add_subplot(111)
        self.pulse_ax.set_facecolor("#0d1520")
        self.pulse_ax.axis("off")
        self.pulse_line, = self.pulse_ax.plot([], [], color=PINK, lw=1.2)
        self.pulse_canvas = FigureCanvasTkAgg(self.pulse_fig, master=hr_pnl)
        self.pulse_canvas.get_tk_widget().pack(fill="x")
        tk.Frame(hr_pnl, bg=PANEL, height=3).pack()

        # Emotion panel
        emo_pnl = tk.Frame(left, bg=PANEL)
        emo_pnl.pack(fill="x", pady=(4, 0))
        emo_hdr = tk.Frame(emo_pnl, bg=PANEL)
        emo_hdr.pack(fill="x", padx=8, pady=(5,2))
        self.emo_title_lbl = tk.Label(emo_hdr, text="EMOTION DETECTION",
                                      font=FONT_SMALL, fg=SUBTEXT, bg=PANEL)
        self.emo_title_lbl.pack(side="left")
        self.dominant_lbl = tk.Label(emo_hdr, text="—",
                                     font=("Courier New", 10, "bold"), fg=ACCENT, bg=PANEL)
        self.dominant_lbl.pack(side="right")

        self.emo_bars   = {}
        self.emo_labels = {}
        for em in ["happy","neutral","surprise","fear","angry","disgust","sad"]:
            row = tk.Frame(emo_pnl, bg=PANEL)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=f"{em:8s}", font=FONT_SMALL,
                     fg=SUBTEXT, bg=PANEL, width=8, anchor="w").pack(side="left")
            bar = tk.Canvas(row, height=9, bg=BORDER, highlightthickness=0, width=250)
            bar.pack(side="left", padx=(2,4))
            lbl = tk.Label(row, text="0%", font=FONT_SMALL, fg=SUBTEXT, bg=PANEL, width=5)
            lbl.pack(side="left")
            self.emo_bars[em]   = bar
            self.emo_labels[em] = lbl
        tk.Frame(emo_pnl, bg=PANEL, height=3).pack()

        # Voice waveform
        wf_pnl = tk.Frame(left, bg=PANEL)
        wf_pnl.pack(fill="x", pady=(4,0))
        self.wave_title_lbl = tk.Label(wf_pnl, text="VOICE WAVEFORM",
                                       font=FONT_SMALL, fg=SUBTEXT, bg=PANEL)
        self.wave_title_lbl.pack(anchor="w", padx=6, pady=2)
        self.wave_fig = Figure(figsize=(4.8, 0.75), facecolor=PANEL)
        self.wave_ax  = self.wave_fig.add_subplot(111)
        self.wave_ax.set_facecolor(PANEL)
        self.wave_ax.axis("off")
        self.wave_line, = self.wave_ax.plot([], [], color=ACCENT, lw=1)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=wf_pnl)
        self.wave_canvas.get_tk_widget().pack(fill="x")

        # RIGHT ───────────────────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG, padx=8)
        right.pack(side="left", fill="both", expand=True)

        # Score panel
        sp = tk.Frame(right, bg=PANEL)
        sp.pack(fill="x", pady=(0,5))
        self.score_lbl = tk.Label(sp, text="--.-", font=FONT_BIG, fg=GREEN, bg=PANEL)
        self.score_lbl.pack(pady=(8,0))
        self.verdict_lbl = tk.Label(sp, text="CLICK CALIBRATE TO BEGIN",
                                    font=("Courier New", 12, "bold"), fg=YELLOW, bg=PANEL)
        self.verdict_lbl.pack(pady=(0,4))
        self.bar_canvas = tk.Canvas(sp, height=12, bg=BORDER, highlightthickness=0)
        self.bar_canvas.pack(fill="x", padx=12, pady=(0,8))

        # Signal breakdown bars (visual representation of 4 signals)
        sig_frame = tk.Frame(right, bg=PANEL)
        sig_frame.pack(fill="x", pady=(0,5))
        tk.Label(sig_frame, text="SIGNAL BREAKDOWN", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", padx=8, pady=(5,2))
        self.sig_bars  = {}
        self.sig_lbls  = {}
        sigs = [("VISUAL",  ACCENT, "vis_s"),
                ("VOICE",   "#a3e635", "voc_s"),
                ("EMOTION", PURPLE,  "emo_s"),
                ("HEART ♥", PINK,    "hr_s")]
        for name, color, key in sigs:
            row = tk.Frame(sig_frame, bg=PANEL)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=f"{name:8s}", font=FONT_SMALL,
                     fg=SUBTEXT, bg=PANEL, width=9, anchor="w").pack(side="left")
            bar = tk.Canvas(row, height=10, bg=BORDER, highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True, padx=(2,4))
            lbl = tk.Label(row, text="0.0", font=FONT_SMALL, fg=color, bg=PANEL, width=5)
            lbl.pack(side="left")
            self.sig_bars[key] = (bar, color)
            self.sig_lbls[key] = lbl
        tk.Frame(sig_frame, bg=PANEL, height=4).pack()

        # Metrics grid
        mf = tk.Frame(right, bg=PANEL)
        mf.pack(fill="x", pady=(0,5))
        tk.Label(mf, text="LIVE METRICS", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).grid(row=0, column=0, columnspan=4,
                                             sticky="w", padx=8, pady=(4,2))
        self._mvars = {}
        items = [
            ("BLINKS/MIN","blinks"), ("EYE SHIFT","eye"),
            ("HEAD MOVE","head"),    ("PITCH Hz","pitch"),
            ("TREMOR","tremor"),     ("PAUSES","pause"),
            ("BPM","bpm"),           ("BPM BASE","bpmbase"),
            ("DOMINANT","dom"),      ("EMO SCR","emoscr"),
            ("HR SCR","hrscr"),      ("FUSED","fused"),
        ]
        for i, (title, key) in enumerate(items):
            r, c = divmod(i, 4)
            tk.Label(mf, text=title, font=FONT_SMALL,
                     fg=SUBTEXT, bg=PANEL).grid(row=r*2+1, column=c, padx=8, pady=(2,0))
            v = tk.StringVar(value="—")
            self._mvars[key] = v
            tk.Label(mf, textvariable=v, font=FONT_MONO,
                     fg=ACCENT, bg=PANEL).grid(row=r*2+2, column=c, padx=8, pady=(0,3))

        # Timeline chart
        cf = tk.Frame(right, bg=PANEL)
        cf.pack(fill="both", expand=True, pady=(0,5))
        self.timeline_lbl = tk.Label(cf, text="SCORE TIMELINE (30s)",
                                     font=FONT_SMALL, fg=SUBTEXT, bg=PANEL)
        self.timeline_lbl.pack(anchor="w", padx=8, pady=2)
        self.chart_fig = Figure(figsize=(5.2, 1.6), facecolor=PANEL)
        self.chart_ax  = self.chart_fig.add_subplot(111)
        self.chart_ax.set_facecolor("#0d1520")
        self.chart_ax.set_ylim(0, 100)
        self.chart_ax.tick_params(colors=SUBTEXT, labelsize=7)
        for s in self.chart_ax.spines.values():
            s.set_edgecolor(BORDER)
        self.chart_line, = self.chart_ax.plot([], [], color=ACCENT, lw=1.5)
        self.chart_fig.tight_layout(pad=0.3)
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=cf)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Question + control bar ────────────────────────────────────────────
        qbar = tk.Frame(right, bg=BG)
        qbar.pack(fill="x", pady=(0,3))

        self.q_var = tk.StringVar(value=self.all_questions[0])
        self.q_menu = tk.OptionMenu(qbar, self.q_var, *self.all_questions)
        self.q_menu.config(bg=PANEL, fg=TEXT, font=FONT_SMALL,
                           activebackground=BORDER, relief="flat", width=30)
        self.q_menu.pack(side="left", padx=(0,3))

        self.add_q_btn = tk.Button(qbar, text="+ ADD", font=FONT_SMALL,
                                   bg="#1a2a1a", fg=GREEN, relief="flat",
                                   padx=5, command=self._add_custom_question)
        self.add_q_btn.pack(side="left", padx=2)

        # Sound toggle
        tk.Checkbutton(qbar, text="🔔", variable=self.sound_on,
                       font=FONT_SMALL, bg=BG, fg=YELLOW,
                       selectcolor=BG, activebackground=BG).pack(side="right")

        ctrl = tk.Frame(right, bg=BG)
        ctrl.pack(fill="x", pady=(0,4))

        self.btn_refs = {}
        for txt_key, cmd, bg, fg in [
            ("calibrate", self._start_calibration, "#1e3a5f", ACCENT),
            ("ask",       self._ask_question,      "#1a3a1a", GREEN),
            ("end",       self._end_question,       "#3a1a1a", RED),
            ("save",      self._save_report,        PANEL,     TEXT),
        ]:
            b = tk.Button(ctrl, text="", command=cmd, font=FONT_SMALL,
                          bg=bg, fg=fg, relief="flat", padx=6)
            b.pack(side="left", padx=2)
            self.btn_refs[txt_key] = b

        # Results
        rf = tk.Frame(right, bg=PANEL)
        rf.pack(fill="x")
        self.results_title_lbl = tk.Label(rf, text="QUESTION RESULTS",
                                          font=FONT_SMALL, fg=SUBTEXT, bg=PANEL)
        self.results_title_lbl.pack(anchor="w", padx=8, pady=(4,2))
        self.results_text = tk.Text(rf, height=4, bg="#0d1520", fg=TEXT,
                                    font=FONT_SMALL, relief="flat", state="disabled")
        self.results_text.pack(fill="x", padx=6, pady=(0,6))

    # ──────────────────────────────────────────────────────────────────────────
    # LANGUAGE
    # ──────────────────────────────────────────────────────────────────────────
    def _on_language_change(self, lang_name):
        self.L = get_lang(lang_name)
        self.all_questions = list(self.L["questions"])
        self._apply_language()

    def _apply_language(self):
        L = self.L
        self.root.title(L["title"])
        self.title_lbl.config(text=L["title"])
        self.hr_title_lbl.config(text=L.get("hr_lbl","HEART RATE (rPPG)"))
        self.emo_title_lbl.config(text=L["emotion_lbl"])
        self.wave_title_lbl.config(text=L["waveform_lbl"])
        self.timeline_lbl.config(text=L["timeline_lbl"])
        self.results_title_lbl.config(text=L["results_lbl"])
        self.add_q_btn.config(text=L["add_q_btn"])
        for key, btn in self.btn_refs.items():
            btn.config(text=L.get(key, key.upper()))
        # Rebuild question menu
        self._rebuild_q_menu()

    def _rebuild_q_menu(self):
        menu = self.q_menu["menu"]
        menu.delete(0, "end")
        for q in self.all_questions:
            menu.add_command(label=q,
                             command=lambda v=q: self.q_var.set(v))
        if self.all_questions:
            self.q_var.set(self.all_questions[0])

    # ──────────────────────────────────────────────────────────────────────────
    # CONTROLS
    # ──────────────────────────────────────────────────────────────────────────
    def _add_custom_question(self):
        q = simpledialog.askstring(
            "Custom Question",
            self.L.get("custom_q_ph", "Type your question:"),
            parent=self.root)
        if q and q.strip():
            self.all_questions.append(q.strip())
            self._rebuild_q_menu()
            self.q_var.set(q.strip())

    def _start_calibration(self):
        self.phase     = "CALIBRATING"
        self.cal_start = time.time()
        self.status_lbl.config(text=self.L["status_cal"], fg=YELLOW)
        self.verdict_lbl.config(text=self.L["cal_prompt"].format(pct=0), fg=YELLOW)

    def _ask_question(self):
        if self.phase not in ("READY", "DETECTING"):
            messagebox.showwarning("!", self.L["not_ready"])
            return
        q = self.q_var.get()
        if not q or q == self.L.get("custom_q_ph", ""):
            messagebox.showwarning("!", self.L["no_question"])
            return
        self.fusion.start_question(q)
        self.phase = "DETECTING"
        self.status_lbl.config(text=self.L["status_det"], fg=RED)

    def _end_question(self):
        if self.phase != "DETECTING":
            return
        self.fusion.end_question()
        self.phase = "READY"
        self.status_lbl.config(text=self.L["status_ready"], fg=GREEN)
        self._refresh_results()

    def _save_report(self):
        path = self.fusion.save_report()
        messagebox.showinfo("✓", self.L["saved_msg"].format(path=path))

    def _refresh_results(self):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        for r in reversed(self.fusion.question_scores[-6:]):
            self.results_text.insert(
                "end", f"[{r['label']:10s} {r['avg_score']:5.1f}]  {r['question']}\n")
        self.results_text.config(state="disabled")

    def _maybe_alert(self, score):
        if (self.sound_on.get()
                and score >= ALERT_THRESHOLD
                and self.phase == "DETECTING"
                and time.time() - self._last_alert > ALERT_COOLDOWN):
            self._last_alert = time.time()
            _beep_alert()

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN TICK
    # ──────────────────────────────────────────────────────────────────────────
    def _tick(self):
        try:
            self._update()
        except Exception as e:
            print(f"[tick] {e}")
        self.root.after(40, self._tick)

    def _update(self):
        now = time.time()

        # ── Webcam ───────────────────────────────────────────────────────────
        frame = self.webcam.process_frame()
        if frame is not None:
            self.emotion.feed_frame(frame)
            # Feed rPPG with landmark info if available
            lm = (self.webcam.face_mesh.process(
                      cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  ).multi_face_landmarks or [None])[0]
            lm_list = lm.landmark if lm else None
            h, w = frame.shape[:2]
            self.rppg.feed_frame(frame, lm_list, w, h)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((480, 340), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.cam_label.config(image=photo)
            self._photo = photo

        # ── rPPG pulse display ────────────────────────────────────────────────
        rsnap = self.rppg.snapshot()
        bpm   = rsnap["bpm"]
        bpm_color = PINK if rsnap["confident"] else SUBTEXT
        bpm_txt   = f"{bpm:.0f} BPM" if bpm > 0 else "-- BPM"
        self.bpm_lbl.config(text=bpm_txt, fg=bpm_color)
        ps = self.rppg.pulse_signal
        if len(ps) > 1:
            x = np.linspace(0, 1, len(ps))
            self.pulse_ax.set_xlim(0, 1)
            self.pulse_ax.set_ylim(-1.2, 1.2)
            self.pulse_line.set_data(x, ps)
            self.pulse_canvas.draw_idle()

        # ── Emotion bars ──────────────────────────────────────────────────────
        esnap    = self.emotion.snapshot()
        emotions = esnap.get("emotions", {})
        dominant = esnap.get("dominant", "neutral")
        dom_conf = esnap.get("confidence", 0.0)
        self.dominant_lbl.config(
            text=f"{dominant.upper()}  {dom_conf:.0f}%",
            fg=EMOTION_COLORS.get(dominant, ACCENT))
        for em, bar in self.emo_bars.items():
            val = emotions.get(em, 0.0)
            w2  = bar.winfo_width()
            bar.delete("all")
            filled = int(w2 * val / 100)
            if filled > 0:
                bar.create_rectangle(0, 0, filled, 9,
                                     fill=EMOTION_COLORS.get(em, ACCENT), outline="")
            self.emo_labels[em].config(text=f"{val:.0f}%")

        # ── Calibration ───────────────────────────────────────────────────────
        if self.phase == "CALIBRATING":
            elapsed = now - self.cal_start
            pct = int(min(elapsed / CALIBRATION_SECS * 100, 100))
            self.verdict_lbl.config(
                text=self.L["cal_prompt"].format(pct=pct), fg=YELLOW)
            self.emotion.collect_calibration()
            self.rppg.collect_calibration()
            if elapsed >= CALIBRATION_SECS:
                self.webcam.calibrate()
                self.voice.calibrate()
                self.emotion.calibrate()
                self.rppg.calibrate()
                self.phase = "READY"
                self.status_lbl.config(text=self.L["status_ready"], fg=GREEN)
                self.verdict_lbl.config(text=self.L["ready_prompt"], fg=GREEN)
            return

        # ── Detection ─────────────────────────────────────────────────────────
        if self.phase in ("READY", "DETECTING"):
            vs  = self.webcam.get_visual_score()
            acs = self.voice.get_voice_score()
            es  = self.emotion.get_emotion_lie_score()
            hrs = self.rppg.get_hr_lie_score()
            self.score = self.fusion.update(vs, acs, es, hrs)
            label, color = score_label(self.score)

            self.score_lbl.config(text=f"{self.score:.1f}", fg=color)
            self.verdict_lbl.config(text=label, fg=color)
            w2 = self.bar_canvas.winfo_width()
            if w2 > 1:
                bw = int(w2 * self.score / 100)
                self.bar_canvas.delete("all")
                self.bar_canvas.create_rectangle(0, 0, bw, 12, fill=color, outline="")

            # Signal breakdown bars
            for key, score_val in [("vis_s", vs), ("voc_s", acs),
                                    ("emo_s", es), ("hr_s",  hrs)]:
                bar, col = self.sig_bars[key]
                bw2 = bar.winfo_width()
                bar.delete("all")
                filled = int(bw2 * score_val / 100)
                if filled > 0:
                    bar.create_rectangle(0, 0, filled, 10, fill=col, outline="")
                self.sig_lbls[key].config(text=f"{score_val:.1f}")

            # Metrics
            ws  = self.webcam.snapshot()
            vss = self.voice.snapshot()
            self._mvars["blinks"].set(f"{ws['blinks_per_min']:.1f}")
            self._mvars["eye"].set(f"{ws['eye_std']:.3f}")
            self._mvars["head"].set(f"{ws['head_avg']:.1f}px")
            self._mvars["pitch"].set(f"{vss['pitch_mean']:.0f}")
            self._mvars["tremor"].set(f"{vss['tremor']:.4f}")
            self._mvars["pause"].set(f"{vss['pause_ratio']:.2f}")
            self._mvars["bpm"].set(f"{rsnap['bpm']:.0f}")
            self._mvars["bpmbase"].set(f"{rsnap['baseline_bpm']:.0f}")
            self._mvars["dom"].set(dominant.upper()[:8])
            self._mvars["emoscr"].set(f"{es:.1f}")
            self._mvars["hrscr"].set(f"{hrs:.1f}")
            self._mvars["fused"].set(f"{self.score:.1f}")

            # Timeline
            times, scores = self.fusion.get_recent_scores(30)
            if len(scores) > 1:
                self.chart_line.set_data(times, scores)
                self.chart_ax.set_xlim(times[0], 0)
                self.chart_canvas.draw_idle()

            # Alert
            self._maybe_alert(self.score)

        # ── Voice waveform ────────────────────────────────────────────────────
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
        self.emotion.stop()
        self.rppg.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = LieDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
