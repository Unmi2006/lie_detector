"""
Session History Browser
Lists all saved JSON/PDF reports in the reports/ folder.
Lets the user view summary of any past session.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os, json, glob, time, subprocess, sys

REPORTS_DIR = "reports"

BG      = "#0a0e17"
PANEL   = "#111827"
BORDER  = "#1e293b"
ACCENT  = "#06b6d4"
TEXT    = "#e2e8f0"
SUBTEXT = "#64748b"
GREEN   = "#22c55e"
YELLOW  = "#f59e0b"
RED     = "#ef4444"

SCORE_COLORS = {
    "TRUTH":     GREEN,
    "UNCERTAIN": "#38bdf8",
    "SUSPECT":   YELLOW,
    "DECEPTION": RED,
}

FONT_MONO  = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_TITLE = ("Courier New", 13, "bold")


def _label(score):
    if score < 25:  return "TRUTH"
    if score < 50:  return "UNCERTAIN"
    if score < 75:  return "SUSPECT"
    return "DECEPTION"


class SessionHistoryWindow:
    def __init__(self, parent=None):
        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.title("Session History")
        self.win.configure(bg=BG)
        self.win.geometry("820x520")
        self.win.resizable(True, True)
        self._build()
        self._load_sessions()

    def _build(self):
        # Header
        hdr = tk.Frame(self.win, bg=BG, pady=6)
        hdr.pack(fill="x", padx=14)
        tk.Label(hdr, text="SESSION HISTORY", font=FONT_TITLE,
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Button(hdr, text="⟳ REFRESH", font=FONT_SMALL,
                  bg=PANEL, fg=ACCENT, relief="flat", padx=6,
                  command=self._load_sessions).pack(side="right")
        tk.Frame(self.win, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self.win, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left: session list
        left = tk.Frame(body, bg=BG, width=280)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="SAVED SESSIONS", font=FONT_SMALL,
                 fg=SUBTEXT, bg=BG).pack(anchor="w", pady=(0,4))

        listframe = tk.Frame(left, bg=PANEL)
        listframe.pack(fill="both", expand=True)
        scrollbar = tk.Scrollbar(listframe)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(listframe, bg=PANEL, fg=TEXT,
                                  font=FONT_SMALL, selectbackground=BORDER,
                                  selectforeground=ACCENT, relief="flat",
                                  yscrollcommand=scrollbar.set,
                                  activestyle="none")
        self.listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Buttons below list
        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(4,0))
        tk.Button(btn_row, text="OPEN PDF", font=FONT_SMALL,
                  bg="#1e3a5f", fg=ACCENT, relief="flat", padx=6,
                  command=self._open_pdf).pack(side="left", padx=(0,4))
        tk.Button(btn_row, text="DELETE", font=FONT_SMALL,
                  bg="#3a1a1a", fg=RED, relief="flat", padx=6,
                  command=self._delete_session).pack(side="left")

        # Right: session detail
        right = tk.Frame(body, bg=BG, padx=10)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="SESSION DETAIL", font=FONT_SMALL,
                 fg=SUBTEXT, bg=BG).pack(anchor="w", pady=(0,4))

        self.detail_frame = tk.Frame(right, bg=PANEL)
        self.detail_frame.pack(fill="both", expand=True)

        # Score big
        self.score_lbl = tk.Label(self.detail_frame, text="—",
                                  font=("Courier New", 36, "bold"),
                                  fg=ACCENT, bg=PANEL)
        self.score_lbl.pack(pady=(18,2))
        self.verdict_lbl = tk.Label(self.detail_frame, text="SELECT A SESSION",
                                    font=("Courier New", 12, "bold"),
                                    fg=SUBTEXT, bg=PANEL)
        self.verdict_lbl.pack()

        # Meta grid
        meta_frame = tk.Frame(self.detail_frame, bg=PANEL)
        meta_frame.pack(fill="x", padx=14, pady=10)
        self._meta_vars = {}
        for i, (title, key) in enumerate([
            ("Date",       "date"),
            ("Questions",  "questions"),
            ("Avg Score",  "avg"),
            ("Peak Score", "peak"),
        ]):
            r, c = divmod(i, 2)
            tk.Label(meta_frame, text=title, font=FONT_SMALL,
                     fg=SUBTEXT, bg=PANEL).grid(row=r*2, column=c, padx=14, sticky="w")
            v = tk.StringVar(value="—")
            self._meta_vars[key] = v
            tk.Label(meta_frame, textvariable=v, font=FONT_MONO,
                     fg=ACCENT, bg=PANEL).grid(row=r*2+1, column=c, padx=14, sticky="w")

        # Questions list
        tk.Label(self.detail_frame, text="QUESTIONS", font=FONT_SMALL,
                 fg=SUBTEXT, bg=PANEL).pack(anchor="w", padx=14, pady=(6,2))
        self.q_text = tk.Text(self.detail_frame, height=8, bg="#0d1520",
                              fg=TEXT, font=FONT_SMALL, relief="flat",
                              state="disabled")
        self.q_text.pack(fill="both", expand=True, padx=14, pady=(0,12))

    def _load_sessions(self):
        self.listbox.delete(0, "end")
        self._sessions = []
        os.makedirs(REPORTS_DIR, exist_ok=True)
        files = sorted(
            glob.glob(os.path.join(REPORTS_DIR, "session_*.json")),
            reverse=True
        )
        for f in files:
            fname = os.path.basename(f)
            self.listbox.insert("end", fname)
            self._sessions.append(f)

        if not files:
            self.listbox.insert("end", "(no sessions saved yet)")

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel or not self._sessions:
            return
        idx = sel[0]
        if idx >= len(self._sessions):
            return
        path = self._sessions[idx]
        try:
            with open(path) as f:
                data = json.load(f)
            summary   = data.get("summary", {})
            avg_score = summary.get("avg_score", 0)
            max_score = summary.get("max_score", 0)
            total_q   = summary.get("total_questions", 0)
            sess_time = data.get("session_time", "—")
            verdict   = _label(avg_score)
            color     = SCORE_COLORS.get(verdict, ACCENT)

            self.score_lbl.config(text=f"{avg_score:.1f}", fg=color)
            self.verdict_lbl.config(text=verdict, fg=color)
            self._meta_vars["date"].set(sess_time)
            self._meta_vars["questions"].set(str(total_q))
            self._meta_vars["avg"].set(f"{avg_score:.1f}")
            self._meta_vars["peak"].set(f"{max_score:.1f}")

            questions = data.get("question_results", [])
            self.q_text.config(state="normal")
            self.q_text.delete("1.0", "end")
            for r in questions:
                lbl = r.get("label", "—")
                sc  = r.get("avg_score", 0)
                q   = r.get("question", "—")
                col = SCORE_COLORS.get(lbl, ACCENT)
                self.q_text.insert("end", f"[{lbl:10s} {sc:5.1f}]  {q}\n")
            self.q_text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load session:\n{e}")

    def _open_pdf(self):
        sel = self.listbox.curselection()
        if not sel or not self._sessions:
            return
        json_path = self._sessions[sel[0]]
        pdf_path  = json_path.replace("session_", "lie_detector_report_").replace(".json", ".pdf")
        if os.path.exists(pdf_path):
            if sys.platform == "win32":
                os.startfile(pdf_path)
            elif sys.platform == "darwin":
                subprocess.call(["open", pdf_path])
            else:
                subprocess.call(["xdg-open", pdf_path])
        else:
            messagebox.showinfo("No PDF", "No PDF found for this session.\nClick SAVE in the main window to generate one.")

    def _delete_session(self):
        sel = self.listbox.curselection()
        if not sel or not self._sessions:
            return
        path = self._sessions[sel[0]]
        if messagebox.askyesno("Delete", f"Delete {os.path.basename(path)}?"):
            try:
                os.remove(path)
            except Exception:
                pass
            self._load_sessions()

    def show(self):
        self.win.mainloop()


if __name__ == "__main__":
    SessionHistoryWindow().show()
