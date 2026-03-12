# 🔍 Lie Detector Pro

A real-time multimodal lie detector using **webcam** (face/eye analysis) and **voice** (audio stress analysis) signals, fused together into a live deception score dashboard.

---

## Features

| Module | Signals Detected |
|--------|-----------------|
| 👁️ Webcam | Blink rate, iris movement, head pose |
| 🎙️ Voice | Pitch, tremor, pause ratio, filler words |
| 🧠 Fusion | Weighted visual + audio score (0–100) |
| 📊 Dashboard | Live waveform, score timeline, metrics HUD |
| 📋 Reports | JSON session reports per question |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> On macOS, install portaudio first: `brew install portaudio`
> On Linux: `sudo apt-get install portaudio19-dev`

### 2. Run
```bash
python dashboard.py
```

---

## How to Use

1. **CALIBRATE** — Click the button, speak normally for 6 seconds (baseline is recorded)
2. **Select a question** from the dropdown
3. **ASK ▶** — Start recording the subject's answer
4. **END ■** — Stop and see the result for that question
5. **SAVE REPORT** — Export session data as JSON

---

## Score Legend

| Score | Label | Meaning |
|-------|-------|---------|
| 0–25  | ✅ TRUTH | Normal physiological response |
| 25–50 | 🔵 UNCERTAIN | Slightly elevated signals |
| 50–75 | 🟠 SUSPECT | Notable stress indicators |
| 75–100| 🔴 DECEPTION | High stress/deception signals |

---

## Project Structure

```
lie_detector_pro/
├── dashboard.py              ← Main GUI entry point
├── modules/
│   ├── webcam_analyzer.py    ← Face/eye analysis (MediaPipe)
│   ├── voice_analyzer.py     ← Audio stress analysis (librosa)
│   └── fusion_engine.py      ← Score fusion + session tracking
├── reports/                  ← Auto-generated JSON reports
└── requirements.txt
```

---

## Tech Stack

- **OpenCV + MediaPipe** — Face mesh, eye tracking, head pose
- **PyAudio + librosa** — Real-time audio capture and pitch analysis
- **SpeechRecognition** — Filler word detection via Google STT
- **Tkinter + Matplotlib** — Live dashboard with charts

---

## ⚠️ Disclaimer

This is an educational/portfolio project. Lie detection via physiological signals is **not scientifically reliable** and should never be used to make real judgments about truthfulness. The tool compares signals against your personal baseline, not absolute thresholds.

---

## Portfolio Notes

This project demonstrates:
- **Real-time computer vision** (OpenCV, MediaPipe facial landmarks)
- **Audio DSP** (pitch estimation, tremor, zero-crossing rate)
- **Multithreaded architecture** (audio stream + analysis in background threads)
- **GUI with embedded live charts** (Tkinter + Matplotlib FigureCanvasTkAgg)
- **Modular design** (cleanly separated analysis, fusion, and UI layers)
- **Session persistence** (JSON report generation)
