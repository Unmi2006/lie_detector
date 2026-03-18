# 🔍 Lie Detector Pro — Ultimate Edition

> A real-time multimodal deception detection system built entirely in Python.  
> Uses your webcam + microphone — **no special hardware needed.**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=flat-square)
![DeepFace](https://img.shields.io/badge/DeepFace-AI-purple?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square)
![License](https://img.shields.io/badge/License-Educational-red?style=flat-square)

---

## 📌 What is this?

Lie Detector Pro analyses 4 physiological signals in real time and fuses them
into a single **deception score (0–100)**. It calibrates to YOUR personal
baseline, so it measures deviation from your normal — not an absolute threshold.

| Signal | Method | Weight |
|--------|--------|--------|
| 👁️ Visual | OpenCV face & eye detection — blink rate, eye shift, head movement | 35% |
| 🎙️ Voice | librosa pitch, tremor, pause ratio via PyAudio mic capture | 30% |
| 😨 Emotion | DeepFace CNN — fear, anger, disgust detection | 20% |
| ❤️ Heart Rate | rPPG CHROM algorithm — pulse from forehead skin color | 15% |

---

## ✨ Features

- 🎯 **Live deception score** — updates every 40ms
- 🧠 **Personal ML model** — Random Forest trains on your own labelled sessions
- 🌐 **5 languages** — English, Hindi, Bengali, Spanish, Chinese
- ☀️ **Dark / Light theme** toggle
- ⛶ **Fullscreen mode**
- ✏️ **Custom questions** — add your own anytime
- 🔔 **Sound alert** — beeps when score crosses threshold
- 📋 **Session history browser** — view and delete past sessions
- 📊 **Excel export** — 4-sheet .xlsx with charts
- 📄 **PDF report** — professional multi-page report with gauge & radar chart
- 📄 **CSV export** — simple timeline data

---

## 🗂️ Project Structure

```
lie_detector_pro/
├── dashboard.py               ← Main GUI (Tkinter) — entry point
├── requirements.txt
├── modules/
│   ├── webcam_analyzer.py     ← OpenCV face/eye detection, blink rate
│   ├── voice_analyzer.py      ← PyAudio + librosa pitch, tremor, pauses
│   ├── emotion_analyzer.py    ← DeepFace background thread (7 emotions)
│   ├── rppg_analyzer.py       ← CHROM rPPG heart rate from webcam
│   ├── fusion_engine.py       ← Weighted score combiner + session history
│   ├── ml_classifier.py       ← RandomForest train / predict / save
│   ├── pdf_report.py          ← ReportLab multi-page PDF generator
│   ├── excel_exporter.py      ← openpyxl 4-sheet workbook
│   ├── session_history.py     ← Tkinter popup session browser
│   └── language_pack.py       ← 5-language string dictionary
├── reports/                   ← Auto-generated PDF, JSON, Excel, CSV
└── models/                    ← Saved ML model (classifier.pkl)
```

---

## ⚙️ Installation

### Step 1 — Clone / download the project

```bash
git clone https://github.com/your-username/lie-detector-pro.git
cd lie-detector-pro
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install "numpy<2" --force-reinstall
pip install protobuf==3.20.3
pip install opencv-python==4.8.1.78
pip install deepface tf-keras
pip install pyaudio librosa SpeechRecognition
pip install Pillow matplotlib scipy
pip install scikit-learn joblib openpyxl reportlab
```

> **Windows only** — if pyaudio fails:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

> **macOS only** — install portaudio first:
> ```bash
> brew install portaudio
> pip install pyaudio
> ```

> **Linux only:**
> ```bash
> sudo apt-get install portaudio19-dev python3-tk
> pip install pyaudio
> ```

### Step 4 — Run

```bash
python dashboard.py
```

---

## 🎮 How to Use

### 1. Calibrate
Click **CALIBRATE** and speak normally for **6 seconds**.  
The system records your personal baseline for all 4 signals.

### 2. Ask a Question
- Select a question from the dropdown, or click **+ ADD** to write your own
- Click **ASK ▶** to start recording the subject's answer
- Click **END ■** when they finish

### 3. Label for ML Training
After END, a popup asks **TRUTH or LIE**.  
Label each question to build your personal training dataset.

### 4. Train the ML Model
After **20+ labelled questions**, click **⚙ TRAIN**.  
The Random Forest model trains on your data and auto-saves to `models/`.

### 5. Export Results
| Button | Output |
|--------|--------|
| **SAVE** | Professional PDF with gauge, radar, timeline charts |
| **📊 EXCEL** | 4-sheet .xlsx workbook with bar/line charts |
| **📄 CSV** | Simple score timeline CSV |
| **📋 HISTORY** | Browse and open all past sessions |

---

## 📊 Score Legend

| Score | Label | What it means |
|-------|-------|---------------|
| 0 – 25 | ✅ TRUTH | Normal — signals at baseline |
| 25 – 50 | 🔵 UNCERTAIN | Slightly elevated — inconclusive |
| 50 – 75 | 🟠 SUSPECT | Notable stress across signals |
| 75 – 100 | 🔴 DECEPTION | High stress — multiple signals elevated |

> Scores are **relative to your personal calibration baseline**, not absolute numbers.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Computer Vision | `opencv-python` — Haar cascade face/eye detection |
| Audio Analysis | `pyaudio`, `librosa`, `SpeechRecognition` |
| Emotion AI | `deepface`, `tf-keras` |
| Heart Rate | `scipy` (bandpass filter + FFT), `numpy` |
| Machine Learning | `scikit-learn` (RandomForest), `joblib` |
| GUI | `tkinter`, `matplotlib` (TkAgg backend), `Pillow` |
| Reporting | `reportlab` (PDF), `openpyxl` (Excel) |

---

## 🌐 Multi-Language Support

Switch language from the dropdown in the top-right corner.

| Language | Questions included |
|----------|--------------------|
| 🇬🇧 English | ✅ |
| 🇮🇳 हिन्दी (Hindi) | ✅ |
| 🇧🇩 বাংলা (Bengali) | ✅ |
| 🇪🇸 Español | ✅ |
| 🇨🇳 中文 (Chinese) | ✅ |

---

## 🔧 Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `protobuf` conflict errors | `pip install protobuf==3.20.3 --force-reinstall` |
| `numpy 2.x` breaks TensorFlow | `pip install "numpy<2" --force-reinstall` |
| Black screen on launch | Camera opened automatically — check camera permissions |
| Webcam not found | Change `open_camera(0)` to `open_camera(1)` in `webcam_analyzer.py` |
| `pyaudio` install fails on Windows | Use `pipwin install pyaudio` instead |
| DeepFace slow on first run | Normal — it downloads the model (~100MB) once |
| `tf-keras` missing | `pip install tf-keras` |

---

## 🚀 Future Upgrade Ideas

- 🌐 **Flask web app** — run in any browser, no installation
- 🎥 **Session video recording** — save webcam + score overlay as .mp4
- 🧠 **Deep learning model** — CNN/LSTM trained on large dataset
- 📱 **Mobile app** — Kivy or React Native version
- 🔌 **Arduino GSR sensor** — add galvanic skin response hardware
- 👥 **Multi-person mode** — track 2 subjects simultaneously
- 😶 **Micro-expression detection** — optical flow for sub-frame expressions

---

## ⚠️ Disclaimer

This is an **educational and portfolio project**.  
Lie detection via physiological signals is **not scientifically validated**
and should **never** be used to make real judgements about a person's
truthfulness. Always use with the subject's full knowledge and consent.
Results are purely for demonstration purposes.

---

## 👨‍💻 Portfolio Highlights

This project demonstrates proficiency in:

- **Real-time computer vision** — OpenCV Haar cascades, live frame processing
- **Audio DSP** — pitch (pyin), tremor, ZCR, bandpass filtering
- **rPPG** — CHROM algorithm, FFT-based BPM estimation
- **Deep learning inference** — DeepFace VGG-Face CNN for emotion
- **Machine learning pipeline** — feature engineering, RandomForest, cross-validation
- **Multithreaded architecture** — background threads for audio, emotion, rPPG
- **GUI development** — Tkinter with embedded Matplotlib charts
- **Report generation** — ReportLab PDF + openpyxl Excel
- **Modular software design** — 11 cleanly separated, independently testable modules
- **Internationalisation** — 5-language runtime switching

---

*Built with ❤️ using Python · OpenCV · DeepFace · scikit-learn*
