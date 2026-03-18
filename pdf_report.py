"""
PDF Report Generator for Lie Detector Pro
Generates a professional multi-page PDF with:
  - Session summary & score gauge
  - Score timeline chart
  - Signal breakdown radar chart
  - Per-question results table & bar chart
Requires: pip install reportlab matplotlib
"""

import os, time, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak
)

# ─── Palette ─────────────────────────────────────────────────────────────────
C_BG      = colors.HexColor("#0a0e17")
C_PANEL   = colors.HexColor("#111827")
C_ACCENT  = colors.HexColor("#06b6d4")
C_TEXT    = colors.HexColor("#e2e8f0")
C_SUBTEXT = colors.HexColor("#64748b")
C_GREEN   = colors.HexColor("#22c55e")
C_BLUE    = colors.HexColor("#38bdf8")
C_ORANGE  = colors.HexColor("#f97316")
C_RED     = colors.HexColor("#ef4444")

SCORE_COLORS = {
    "TRUTH":     "#22c55e",
    "UNCERTAIN": "#38bdf8",
    "SUSPECT":   "#f97316",
    "DECEPTION": "#ef4444",
}

def _label(score):
    if score < 25:  return "TRUTH"
    if score < 50:  return "UNCERTAIN"
    if score < 75:  return "SUSPECT"
    return "DECEPTION"

def _hex_color(score):
    return SCORE_COLORS.get(_label(score), "#06b6d4")

def _fig_to_image(fig, w_mm, h_mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return RLImage(buf, width=w_mm*mm, height=h_mm*mm)


# ─── Charts ──────────────────────────────────────────────────────────────────
def _timeline_chart(history, w=160, h=52):
    fig, ax = plt.subplots(figsize=(w/25.4, h/25.4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#0d1520")
    if history:
        times  = [t for t,_ in history]
        scores = [s for _,s in history]
        t0 = times[0]
        rel = [t-t0 for t in times]
        for i in range(len(scores)-1):
            c = SCORE_COLORS[_label(scores[i])]
            ax.plot(rel[i:i+2], scores[i:i+2], color=c, lw=2)
        ax.fill_between(rel, scores, alpha=0.1, color="#06b6d4")
        for lo,hi,col in [(0,25,"#22c55e"),(25,50,"#38bdf8"),(50,75,"#f97316"),(75,100,"#ef4444")]:
            ax.axhspan(lo, hi, alpha=0.05, color=col)
    ax.set_ylim(0,100)
    ax.set_xlabel("Time (s)", color="#64748b", fontsize=7)
    ax.set_ylabel("Score",    color="#64748b", fontsize=7)
    ax.tick_params(colors="#64748b", labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor("#1e293b")
    ax.set_title("Score Timeline", color="#e2e8f0", fontsize=9, pad=4)
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, w, h)


def _radar_chart(signals, w=80, h=80):
    labels = ["Visual","Voice","Emotion","Heart Rate"]
    vals   = [signals.get("avg_visual",0), signals.get("avg_voice",0),
              signals.get("avg_emotion",0), signals.get("avg_hr",0)]
    N      = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    v_plot = vals + [vals[0]]
    a_plot = angles + [angles[0]]
    fig, ax = plt.subplots(figsize=(w/25.4, h/25.4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#0d1520")
    ax.plot(a_plot, v_plot, color="#06b6d4", lw=2)
    ax.fill(a_plot, v_plot, color="#06b6d4", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#e2e8f0", fontsize=7)
    ax.set_ylim(0,100)
    ax.set_yticks([25,50,75])
    ax.set_yticklabels(["25","50","75"], color="#64748b", fontsize=6)
    ax.grid(color="#1e293b", lw=0.8)
    ax.spines["polar"].set_color("#1e293b")
    ax.set_title("Signal Breakdown", color="#e2e8f0", fontsize=9, pad=10)
    fig.tight_layout(pad=0.3)
    return _fig_to_image(fig, w, h)


def _question_bar(questions, w=160, h=55):
    if not questions: return None
    labels = [f"Q{i+1}" for i in range(len(questions))]
    scores = [r["avg_score"] for r in questions]
    bcolors= [SCORE_COLORS.get(r["label"],"#06b6d4") for r in questions]
    fig, ax = plt.subplots(figsize=(w/25.4, h/25.4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#0d1520")
    bars = ax.barh(labels, scores, color=bcolors, height=0.5, edgecolor="none")
    for bar, sc in zip(bars, scores):
        ax.text(sc+1, bar.get_y()+bar.get_height()/2,
                f"{sc:.1f}", va="center", color="#e2e8f0", fontsize=7)
    ax.set_xlim(0,108)
    ax.set_xlabel("Lie Score", color="#64748b", fontsize=7)
    ax.tick_params(colors="#64748b", labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor("#1e293b")
    for x,col in [(25,"#22c55e"),(50,"#38bdf8"),(75,"#f97316")]:
        ax.axvline(x, color=col, lw=0.8, ls="--", alpha=0.5)
    ax.set_title("Score Per Question", color="#e2e8f0", fontsize=9, pad=4)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.5)
    return _fig_to_image(fig, w, h)


def _gauge(score, w=80, h=52):
    fig, ax = plt.subplots(figsize=(w/25.4, h/25.4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.set_aspect("equal")
    for lo,hi,col in [(0,.25,"#22c55e"),(.25,.5,"#38bdf8"),(.5,.75,"#f97316"),(.75,1,"#ef4444")]:
        theta = np.linspace(np.pi*(1-hi), np.pi*(1-lo), 60)
        ro,ri = 1.0, 0.6
        x = np.concatenate([ro*np.cos(theta), ri*np.cos(theta[::-1])])
        y = np.concatenate([ro*np.sin(theta), ri*np.sin(theta[::-1])])
        ax.fill(x, y, color=col, alpha=0.4)
    ang = np.pi*(1 - score/100)
    ax.plot([0, 0.75*np.cos(ang)],[0, 0.75*np.sin(ang)],
            color=_hex_color(score), lw=2.5, solid_capstyle="round")
    ax.add_patch(plt.Circle((0,0), 0.07, color="#e2e8f0", zorder=5))
    lbl = _label(score)
    col = _hex_color(score)
    ax.text(0,-0.22, f"{score:.1f}", ha="center", va="center",
            color=col, fontsize=13, fontweight="bold", fontfamily="monospace")
    ax.text(0,-0.42, lbl, ha="center", va="center",
            color=col, fontsize=8, fontfamily="monospace")
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.6,1.1); ax.axis("off")
    fig.tight_layout(pad=0.1)
    return _fig_to_image(fig, w, h)


# ─── Main ─────────────────────────────────────────────────────────────────────
def generate_pdf_report(report_data: dict, output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts    = time.strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(output_dir, f"lie_detector_report_{ts}.pdf")

    doc = SimpleDocTemplate(fname, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm,  bottomMargin=15*mm)

    title_s = ParagraphStyle("T", fontSize=20, fontName="Courier-Bold",
        textColor=C_ACCENT, alignment=TA_CENTER, spaceAfter=2)
    sub_s   = ParagraphStyle("S", fontSize=9,  fontName="Courier",
        textColor=C_SUBTEXT, alignment=TA_CENTER, spaceAfter=10)
    h1_s    = ParagraphStyle("H1", fontSize=12, fontName="Courier-Bold",
        textColor=C_ACCENT, spaceBefore=8, spaceAfter=4)
    disc_s  = ParagraphStyle("D", fontSize=7, fontName="Courier",
        textColor=C_SUBTEXT, alignment=TA_CENTER, spaceBefore=4)

    summary   = report_data.get("summary", {})
    signals   = report_data.get("signals", {})
    questions = report_data.get("question_results", [])
    history   = report_data.get("score_history", [])
    sess_time = report_data.get("session_time", "—")
    avg_score = summary.get("avg_score", 0)
    max_score = summary.get("max_score", 0)
    total_q   = summary.get("total_questions", 0)
    verdict   = _label(avg_score)

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("LIE DETECTOR PRO", title_s))
    story.append(Paragraph("Session Analysis Report", sub_s))
    story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceAfter=8))

    # Meta table
    meta = [
        ["Session Date", sess_time],
        ["Questions Asked", str(total_q)],
        ["Average Score",  f"{avg_score:.1f} / 100"],
        ["Peak Score",     f"{max_score:.1f} / 100"],
        ["Overall Verdict", verdict],
    ]
    mt = Table(meta, colWidths=[55*mm, 110*mm])
    verdict_color = colors.HexColor(SCORE_COLORS.get(verdict, "#06b6d4"))
    mt.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Courier"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("FONTNAME",    (1,4),(1,4),   "Courier-Bold"),
        ("TEXTCOLOR",   (0,0),(0,-1),  C_SUBTEXT),
        ("TEXTCOLOR",   (1,0),(1,-1),  C_TEXT),
        ("TEXTCOLOR",   (1,4),(1, 4),  verdict_color),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),
         [colors.HexColor("#111827"), colors.HexColor("#0d1520")]),
        ("GRID",        (0,0),(-1,-1), 0.4, C_PANEL),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(mt)
    story.append(Spacer(1, 6*mm))

    # Gauge + Radar
    gauge_img = _gauge(avg_score)
    radar_img = _radar_chart(signals)
    sr = Table([[gauge_img, radar_img]], colWidths=[90*mm, 90*mm])
    sr.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(sr)
    story.append(Spacer(1, 4*mm))

    # Signal averages table
    story.append(Paragraph("SIGNAL AVERAGES", h1_s))
    sig_rows = [["Signal","Avg Score","Weight","Contribution"],
        ["Visual  (eye/blink/head)", f"{signals.get('avg_visual',0):.1f}",  "35%", f"{signals.get('avg_visual',0)*0.35:.1f}"],
        ["Voice   (pitch/tremor)",   f"{signals.get('avg_voice',0):.1f}",   "30%", f"{signals.get('avg_voice',0)*0.30:.1f}"],
        ["Emotion (DeepFace)",       f"{signals.get('avg_emotion',0):.1f}", "20%", f"{signals.get('avg_emotion',0)*0.20:.1f}"],
        ["Heart Rate (rPPG)",        f"{signals.get('avg_hr',0):.1f}",      "15%", f"{signals.get('avg_hr',0)*0.15:.1f}"],
    ]
    st2 = Table(sig_rows, colWidths=[80*mm,30*mm,25*mm,35*mm])
    st2.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), C_ACCENT),
        ("TEXTCOLOR",   (0,0),(-1,0), colors.HexColor("#0a0e17")),
        ("FONTNAME",    (0,0),(-1,0), "Courier-Bold"),
        ("FONTNAME",    (0,1),(-1,-1),"Courier"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#111827"),colors.HexColor("#0d1520")]),
        ("TEXTCOLOR",   (0,1),(-1,-1), C_TEXT),
        ("GRID",        (0,0),(-1,-1), 0.4, C_PANEL),
        ("LEFTPADDING", (0,0),(-1,-1), 6),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("ALIGN",       (1,0),(-1,-1),"CENTER"),
    ]))
    story.append(st2)

    # ── PAGE 2 ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("SCORE TIMELINE", h1_s))
    story.append(_timeline_chart(history))
    story.append(Spacer(1, 5*mm))

    if questions:
        story.append(Paragraph("SCORE PER QUESTION", h1_s))
        qbar = _question_bar(questions)
        if qbar: story.append(qbar)
        story.append(Spacer(1, 5*mm))

        story.append(Paragraph("QUESTION DETAILS", h1_s))
        q_rows = [["#","Question","Score","Verdict"]]
        for i,r in enumerate(questions):
            q = r["question"]
            if len(q) > 58: q = q[:58]+"..."
            q_rows.append([str(i+1), q, f"{r['avg_score']:.1f}", r["label"]])
        qt = Table(q_rows, colWidths=[10*mm,110*mm,22*mm,28*mm])
        ts_cmds = [
            ("BACKGROUND",  (0,0),(-1,0), C_ACCENT),
            ("TEXTCOLOR",   (0,0),(-1,0), colors.HexColor("#0a0e17")),
            ("FONTNAME",    (0,0),(-1,0), "Courier-Bold"),
            ("FONTNAME",    (0,1),(-1,-1),"Courier"),
            ("FONTSIZE",    (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#111827"),colors.HexColor("#0d1520")]),
            ("TEXTCOLOR",   (0,1),(-1,-1), C_TEXT),
            ("GRID",        (0,0),(-1,-1), 0.4, C_PANEL),
            ("LEFTPADDING", (0,0),(-1,-1), 5),
            ("TOPPADDING",  (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("ALIGN",       (0,0),(0,-1),"CENTER"),
            ("ALIGN",       (2,0),(-1,-1),"CENTER"),
        ]
        for i,r in enumerate(questions):
            c = colors.HexColor(SCORE_COLORS.get(r["label"],"#06b6d4"))
            ts_cmds += [("TEXTCOLOR",(3,i+1),(3,i+1),c),
                        ("FONTNAME", (3,i+1),(3,i+1),"Courier-Bold")]
        qt.setStyle(TableStyle(ts_cmds))
        story.append(qt)

    # Footer
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_SUBTEXT))
    story.append(Paragraph(
        "Lie Detector Pro — Educational/Portfolio Project. "
        "Results are not scientifically validated.",
        disc_s))

    doc.build(story)
    return fname
