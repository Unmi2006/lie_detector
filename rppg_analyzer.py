"""
rPPG Heart Rate Analyzer
Estimates heart rate from subtle skin color changes in the forehead region.
Uses the CHROM (Chrominance-based) algorithm — no hardware needed.

Reference: De Haan & Jeanne (2013), IEEE TBME
"""

import cv2
import numpy as np
import threading
import time
import collections
from scipy.signal import butter, filtfilt, find_peaks

# ─── Config ──────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 30        # webcam FPS
WINDOW_SECS    = 10        # rolling window for BPM estimation
BPM_MIN        = 45
BPM_MAX        = 180
BUTTERWORTH_LO = 0.7       # Hz  (~42 BPM)
BUTTERWORTH_HI = 3.5       # Hz  (~210 BPM)

# Forehead ROI as fraction of face bounding box
FOREHEAD_TOP    = 0.10
FOREHEAD_BOTTOM = 0.35
FOREHEAD_LEFT   = 0.25
FOREHEAD_RIGHT  = 0.75

# MediaPipe forehead landmark indices (approximate top of face)
FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389,
                       356, 454, 323, 361, 288, 397, 365,
                       379, 378, 400, 377, 152, 148, 176,
                       149, 150, 136, 172, 58,  132, 93,
                       234, 127, 162, 21,  54,  103, 67, 109]


def _bandpass(signal, lo, hi, fs):
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    lo_n, hi_n = lo / nyq, hi / nyq
    lo_n = max(0.001, min(lo_n, 0.999))
    hi_n = max(0.001, min(hi_n, 0.999))
    if lo_n >= hi_n:
        return signal
    b, a = butter(3, [lo_n, hi_n], btype="band")
    if len(signal) < max(len(a), len(b)) * 3:
        return signal
    return filtfilt(b, a, signal)


class RPPGAnalyzer:
    def __init__(self):
        self._lock        = threading.Lock()
        self.running      = False

        # RGB channel buffers
        max_buf = SAMPLE_RATE * WINDOW_SECS
        self._r_buf = collections.deque(maxlen=max_buf)
        self._g_buf = collections.deque(maxlen=max_buf)
        self._b_buf = collections.deque(maxlen=max_buf)
        self._t_buf = collections.deque(maxlen=max_buf)

        # Results
        self.bpm           = 0.0
        self.bpm_confident = False
        self.pulse_signal  = np.zeros(64)   # for waveform display
        self.signal_quality = 0.0           # 0-1

        # Baseline
        self.baseline_bpm  = 70.0
        self.calibrated    = False
        self._cal_buf      = []

        # Stress contribution
        self._bpm_history  = collections.deque(maxlen=30)

    # ── Control ───────────────────────────────────────────────────────────────
    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    # ── Feed a face ROI frame ─────────────────────────────────────────────────
    def feed_frame(self, frame_bgr, face_landmarks=None, frame_w=640, frame_h=480):
        """
        Call every webcam frame.
        Extracts forehead ROI and appends mean RGB to buffers.
        """
        if not self.running:
            return

        roi = self._extract_roi(frame_bgr, face_landmarks, frame_w, frame_h)
        if roi is None or roi.size == 0:
            return

        # Mean RGB of ROI
        mean_b = float(roi[:, :, 0].mean())
        mean_g = float(roi[:, :, 1].mean())
        mean_r = float(roi[:, :, 2].mean())

        with self._lock:
            self._r_buf.append(mean_r)
            self._g_buf.append(mean_g)
            self._b_buf.append(mean_b)
            self._t_buf.append(time.time())

        # Run estimation every 15 new samples
        if len(self._r_buf) % 15 == 0:
            self._estimate_bpm()

    def _extract_roi(self, frame, landmarks, w, h):
        """Extract forehead region from frame."""
        try:
            if landmarks:
                # Use MediaPipe landmarks to get tight forehead box
                pts = np.array([
                    [int(landmarks[i].x * w), int(landmarks[i].y * h)]
                    for i in FOREHEAD_LANDMARKS
                    if i < len(landmarks)
                ])
                if len(pts) < 4:
                    return self._fallback_roi(frame)
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                fh = h * 0.15
                y2 = int(y1 + fh)
                y1 = max(0, y1)
                x1 = max(0, int(x1 + (x2-x1)*0.15))
                x2 = min(w, int(x2 - (x2-x1)*0.15))
                roi = frame[y1:y2, x1:x2]
                return roi if roi.size > 0 else self._fallback_roi(frame)
            else:
                return self._fallback_roi(frame)
        except Exception:
            return self._fallback_roi(frame)

    def _fallback_roi(self, frame):
        """Use top-center 20% of frame as forehead proxy."""
        h, w = frame.shape[:2]
        y1, y2 = int(h * 0.08), int(h * 0.28)
        x1, x2 = int(w * 0.30), int(w * 0.70)
        return frame[y1:y2, x1:x2]

    # ── CHROM rPPG algorithm ──────────────────────────────────────────────────
    def _estimate_bpm(self):
        with self._lock:
            r = np.array(self._r_buf, dtype=np.float64)
            g = np.array(self._g_buf, dtype=np.float64)
            b = np.array(self._b_buf, dtype=np.float64)

        n = len(r)
        if n < SAMPLE_RATE * 3:   # need at least 3 seconds
            return

        # Normalise each channel
        def norm(x):
            m = x.mean()
            return (x / m) - 1.0 if m > 1e-6 else x

        Rn, Gn, Bn = norm(r), norm(g), norm(b)

        # CHROM: Xs = 3R - 2G,  Ys = 1.5R + G - 1.5B
        Xs = 3 * Rn - 2 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn

        # Alpha to remove specular noise
        std_x = Xs.std() + 1e-9
        std_y = Ys.std() + 1e-9
        alpha  = std_x / std_y
        pulse  = Xs - alpha * Ys

        # Bandpass filter
        pulse_f = _bandpass(pulse, BUTTERWORTH_LO, BUTTERWORTH_HI, SAMPLE_RATE)

        # FFT-based frequency estimation
        fft_vals = np.abs(np.fft.rfft(pulse_f * np.hanning(len(pulse_f))))
        freqs    = np.fft.rfftfreq(len(pulse_f), d=1.0 / SAMPLE_RATE)

        mask = (freqs >= BUTTERWORTH_LO) & (freqs <= BUTTERWORTH_HI)
        if not mask.any():
            return

        peak_idx  = np.argmax(fft_vals[mask])
        peak_freq = freqs[mask][peak_idx]
        bpm_est   = peak_freq * 60.0

        # Signal quality: peak-to-mean ratio in band
        band_power   = fft_vals[mask]
        quality      = float(band_power[peak_idx] / (band_power.mean() + 1e-9))
        quality_norm = float(np.clip((quality - 1) / 9.0, 0, 1))

        if BPM_MIN <= bpm_est <= BPM_MAX:
            # Smooth with history
            self._bpm_history.append(bpm_est)
            smoothed = float(np.median(list(self._bpm_history)))

            with self._lock:
                self.bpm            = smoothed
                self.bpm_confident  = quality_norm > 0.3
                self.signal_quality = quality_norm
                # Store last 64 samples of pulse signal for display
                disp = pulse_f[-64:] if len(pulse_f) >= 64 else pulse_f
                mx   = np.abs(disp).max()
                self.pulse_signal = disp / (mx + 1e-9)

            if self.calibrated:
                self._cal_buf.append(smoothed)

    # ── Calibration ───────────────────────────────────────────────────────────
    def collect_calibration(self):
        if self.bpm > 0:
            self._cal_buf.append(self.bpm)

    def calibrate(self):
        if self._cal_buf:
            self.baseline_bpm = float(np.median(self._cal_buf))
        else:
            self.baseline_bpm = 70.0
        self._cal_buf  = []
        self.calibrated = True

    # ── Lie score contribution ────────────────────────────────────────────────
    def get_hr_lie_score(self):
        """
        Returns 0-100 based on heart rate elevation above baseline.
        Stress typically raises HR by 10-30 BPM.
        """
        if not self.calibrated or self.bpm <= 0:
            return 0.0
        delta = self.bpm - self.baseline_bpm
        # 0 at baseline, 100 at +30 BPM above baseline
        score = float(np.clip(delta / 30.0 * 100, 0, 100))
        return score

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self):
        with self._lock:
            return {
                "bpm":          round(self.bpm, 1),
                "baseline_bpm": round(self.baseline_bpm, 1),
                "confident":    self.bpm_confident,
                "quality":      round(self.signal_quality, 2),
                "hr_score":     round(self.get_hr_lie_score(), 1),
            }
