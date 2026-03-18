"""
ML Classifier — Lie Detector Pro
Trains a Random Forest on your own session data.
Features: visual, voice, emotion, hr scores + derived features
Labels: manually set per question (truth=0, lie=1)

Workflow:
  1. collect_sample() called every tick during detection
  2. label_last_question(label) called after END to tag it
  3. train() called when enough labelled data exists
  4. predict(features) returns probability of deception
"""

import os
import json
import time
import numpy as np
import collections

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[MLClassifier] scikit-learn not installed. Run: pip install scikit-learn joblib")

MIN_SAMPLES_TO_TRAIN = 20   # minimum labelled samples before training
MODEL_DIR = "models"


class MLClassifier:
    def __init__(self):
        self.available   = SKLEARN_AVAILABLE
        self.trained     = False
        self.pipeline    = None
        self.accuracy    = 0.0
        self.n_samples   = 0

        # Raw feature buffer (filled every tick)
        self._sample_buf  = []   # list of feature dicts
        self._labels      = []   # 0=truth, 1=lie per question block
        self._q_blocks    = []   # list of (start_idx, end_idx, label)
        self._current_start = None

        # Prediction smoothing
        self._pred_history = collections.deque(maxlen=10)
        self.deception_prob = 0.0   # latest smoothed prediction

        # Load existing model if present
        self._load_model()

    # ── Sample collection ─────────────────────────────────────────────────────
    def start_question(self):
        self._current_start = len(self._sample_buf)

    def collect_sample(self, visual: float, voice: float,
                       emotion: float, hr: float,
                       blinks: float, eye_std: float,
                       head_avg: float, pitch: float,
                       tremor: float, pause: float,
                       bpm: float, dominant_emotion: str = "neutral"):
        """Call every tick during detection to accumulate features."""
        features = {
            "visual": visual, "voice": voice,
            "emotion": emotion, "hr": hr,
            "blinks": blinks, "eye_std": eye_std,
            "head_avg": head_avg, "pitch": pitch,
            "tremor": tremor, "pause": pause,
            "bpm": bpm,
            # Derived
            "stress_combo": (visual + voice + emotion) / 3.0,
            "face_stress":  (blinks * 0.4 + eye_std * 100 * 0.3 + head_avg * 0.3),
            "voice_stress": (pitch / 300.0 * 50 + tremor * 1000 + pause * 30),
            "is_fear":      1.0 if dominant_emotion == "fear"    else 0.0,
            "is_angry":     1.0 if dominant_emotion == "angry"   else 0.0,
            "is_disgust":   1.0 if dominant_emotion == "disgust" else 0.0,
        }
        self._sample_buf.append(features)

    def label_last_question(self, label: int):
        """
        label: 0 = truth, 1 = lie
        Call after end_question() to tag the last question block.
        """
        if self._current_start is None:
            return
        end = len(self._sample_buf)
        if end > self._current_start:
            self._q_blocks.append((self._current_start, end, label))
            self.n_samples += 1
        self._current_start = None

    # ── Training ──────────────────────────────────────────────────────────────
    def can_train(self):
        return self.available and self.n_samples >= MIN_SAMPLES_TO_TRAIN

    def train(self):
        """Train on all labelled question blocks. Returns accuracy string."""
        if not self.can_train():
            return f"Need {MIN_SAMPLES_TO_TRAIN} labelled samples (have {self.n_samples})"

        X, y = [], []
        for start, end, label in self._q_blocks:
            block = self._sample_buf[start:end]
            if not block:
                continue
            # Aggregate features per question block
            keys = list(block[0].keys())
            agg  = {}
            for k in keys:
                vals = [s[k] for s in block]
                agg[f"{k}_mean"] = np.mean(vals)
                agg[f"{k}_std"]  = np.std(vals)
                agg[f"{k}_max"]  = np.max(vals)
            X.append(list(agg.values()))
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Build pipeline
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=100, max_depth=6,
                random_state=42, class_weight="balanced"
            ))
        ])

        # Cross-val accuracy
        if len(set(y)) > 1 and len(y) >= 4:
            cv_scores    = cross_val_score(self.pipeline, X, y, cv=min(5, len(y)//2))
            self.accuracy = float(cv_scores.mean())
        else:
            self.accuracy = 0.0

        self.pipeline.fit(X, y)
        self.trained = True
        self._save_model()
        return f"Trained on {len(y)} samples. CV accuracy: {self.accuracy*100:.1f}%"

    # ── Prediction (real-time) ────────────────────────────────────────────────
    def predict_live(self, features: dict) -> float:
        """
        Returns deception probability 0-100 for a single feature snapshot.
        Falls back to 0 if not trained.
        """
        if not self.trained or self.pipeline is None:
            return 0.0
        try:
            # Use mean/std/max of a short rolling window from _sample_buf
            window = self._sample_buf[-15:] if len(self._sample_buf) >= 15 \
                     else self._sample_buf
            if not window:
                return 0.0
            keys = list(window[0].keys())
            agg  = {}
            for k in keys:
                vals = [s[k] for s in window]
                agg[f"{k}_mean"] = np.mean(vals)
                agg[f"{k}_std"]  = np.std(vals)
                agg[f"{k}_max"]  = np.max(vals)
            X = np.array([list(agg.values())], dtype=np.float32)
            prob = float(self.pipeline.predict_proba(X)[0][1]) * 100
            self._pred_history.append(prob)
            self.deception_prob = float(np.mean(self._pred_history))
            return self.deception_prob
        except Exception as e:
            return 0.0

    # ── Persist ───────────────────────────────────────────────────────────────
    def _save_model(self):
        if not self.available or self.pipeline is None:
            return
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(MODEL_DIR, "classifier.pkl"))
        meta = {"accuracy": self.accuracy, "n_samples": self.n_samples,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _load_model(self):
        if not self.available:
            return
        path = os.path.join(MODEL_DIR, "classifier.pkl")
        meta_path = os.path.join(MODEL_DIR, "meta.json")
        if os.path.exists(path):
            try:
                self.pipeline = joblib.load(path)
                self.trained  = True
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self.accuracy  = meta.get("accuracy", 0.0)
                    self.n_samples = meta.get("n_samples", 0)
            except Exception:
                pass

    def export_csv(self, path="reports"):
        """Export all labelled samples to CSV for manual analysis."""
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, f"training_data_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        rows  = []
        for start, end, label in self._q_blocks:
            block = self._sample_buf[start:end]
            for s in block:
                row = dict(s)
                row["label"] = label
                rows.append(row)
        if rows:
            import csv
            with open(fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        return fname

    def snapshot(self):
        return {
            "trained":        self.trained,
            "n_samples":      self.n_samples,
            "accuracy":       round(self.accuracy * 100, 1),
            "deception_prob": round(self.deception_prob, 1),
            "can_train":      self.can_train(),
        }
