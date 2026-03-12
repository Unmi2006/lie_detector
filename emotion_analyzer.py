"""
Emotion Analyzer Module
Uses DeepFace to detect emotions from webcam frames.
Runs in a background thread to avoid blocking the UI.
"""

import threading
import time
import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[EmotionAnalyzer] DeepFace not installed. Run: pip install deepface")

# Emotions that contribute to deception stress score
STRESS_EMOTIONS = {"fear", "angry", "disgust", "surprise"}
CALM_EMOTIONS   = {"happy", "neutral"}

# Weight of each emotion toward lie score (0-1)
EMOTION_WEIGHTS = {
    "angry":   0.85,
    "fear":    1.00,
    "disgust": 0.70,
    "surprise":0.55,
    "sad":     0.40,
    "neutral": 0.00,
    "happy":   0.00,
}


class EmotionAnalyzer:
    def __init__(self, interval=0.8):
        """
        interval: seconds between DeepFace analyses (keep >=0.5 for performance)
        """
        self.interval      = interval
        self.running       = False
        self._thread       = None
        self._lock         = threading.Lock()
        self._latest_frame = None

        # Latest results
        self.emotions      = {}          # {"happy": 92.1, "fear": 3.2, ...}
        self.dominant      = "neutral"
        self.dominant_conf = 0.0
        self.emotion_score = 0.0        # 0-100 stress contribution
        self.available     = DEEPFACE_AVAILABLE

        # Baseline (set during calibration)
        self._baseline_score   = 0.0
        self._calibration_buf  = []
        self.calibrated        = False

        # History for smoothing
        self._score_history = []

    # ── Control ──────────────────────────────────────────────────────────────
    def start(self):
        if not self.available:
            return
        self.running  = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    # ── Feed frames from webcam ───────────────────────────────────────────────
    def feed_frame(self, bgr_frame):
        """Call this every webcam tick to keep the analyzer up to date."""
        with self._lock:
            self._latest_frame = bgr_frame.copy()

    # ── Background analysis loop ──────────────────────────────────────────────
    def _loop(self):
        while self.running:
            time.sleep(self.interval)
            with self._lock:
                frame = self._latest_frame
            if frame is None:
                continue
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )
                # result can be a list or dict depending on version
                if isinstance(result, list):
                    result = result[0]

                raw = result.get("emotion", {})
                dominant = result.get("dominant_emotion", "neutral")

                # Normalise to 0-100
                total = sum(raw.values()) or 1
                normed = {k: v / total * 100 for k, v in raw.items()}

                # Compute stress score
                score = sum(
                    normed.get(em, 0) * w
                    for em, w in EMOTION_WEIGHTS.items()
                )
                score = float(np.clip(score, 0, 100))

                self._score_history.append(score)
                if len(self._score_history) > 5:
                    self._score_history.pop(0)
                smoothed = float(np.mean(self._score_history))

                with self._lock:
                    self.emotions      = normed
                    self.dominant      = dominant
                    self.dominant_conf = normed.get(dominant, 0.0)
                    self.emotion_score = smoothed

            except Exception as e:
                # Face not detected or other error — keep last values
                pass

    # ── Calibration ───────────────────────────────────────────────────────────
    def collect_calibration(self):
        """Call every tick during calibration phase."""
        self._calibration_buf.append(self.emotion_score)

    def calibrate(self):
        if self._calibration_buf:
            self._baseline_score = float(np.mean(self._calibration_buf))
        self._calibration_buf = []
        self.calibrated = True

    # ── Score relative to baseline ────────────────────────────────────────────
    def get_emotion_lie_score(self):
        """Returns 0-100 contribution to lie score."""
        if not self.calibrated or not self.available:
            return self.emotion_score
        delta = self.emotion_score - self._baseline_score
        return float(np.clip(delta * 1.5 + self._baseline_score, 0, 100))

    # ── Snapshot for report / metrics ─────────────────────────────────────────
    def snapshot(self):
        with self._lock:
            return {
                "dominant":      self.dominant,
                "confidence":    round(self.dominant_conf, 1),
                "emotion_score": round(self.emotion_score, 1),
                "lie_contrib":   round(self.get_emotion_lie_score(), 1),
                "emotions":      {k: round(v, 1) for k, v in self.emotions.items()},
            }
