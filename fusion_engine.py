"""
Fusion Engine — 4-signal weighted score
Visual 35% | Voice 30% | Emotion 20% | Heart Rate 15%
"""

import time
import json
import os
import collections
import numpy as np


VISUAL_WEIGHT   = 0.35
VOICE_WEIGHT    = 0.30
EMOTION_WEIGHT  = 0.20
HR_WEIGHT       = 0.15

LABELS = [
    (25,  "TRUTH",     "#22c55e"),
    (50,  "UNCERTAIN", "#38bdf8"),
    (75,  "SUSPECT",   "#f97316"),
    (101, "DECEPTION", "#ef4444"),
]


def score_label(score):
    for threshold, label, color in LABELS:
        if score < threshold:
            return label, color
    return "DECEPTION", "#ef4444"


class FusionEngine:
    def __init__(self, history_len=300):
        self.history         = collections.deque(maxlen=history_len)
        self.question_scores = []
        self._current_question = None
        self._question_start   = None
        self._question_buffer  = []

    def update(self, visual_score: float, voice_score: float,
               emotion_score: float = 0.0, hr_score: float = 0.0) -> float:
        fused = (VISUAL_WEIGHT  * visual_score
               + VOICE_WEIGHT   * voice_score
               + EMOTION_WEIGHT * emotion_score
               + HR_WEIGHT      * hr_score)
        fused = float(np.clip(fused, 0, 100))
        self.history.append((time.time(), fused))
        if self._current_question is not None:
            self._question_buffer.append(fused)
        return fused

    def start_question(self, question: str):
        self._current_question = question
        self._question_start   = time.time()
        self._question_buffer  = []

    def end_question(self):
        if self._current_question and self._question_buffer:
            avg = float(np.mean(self._question_buffer))
            label, color = score_label(avg)
            self.question_scores.append({
                "question":  self._current_question,
                "avg_score": round(avg, 1),
                "label":     label,
                "color":     color,
                "timestamp": self._question_start,
            })
        self._current_question = None
        self._question_buffer  = []

    def get_recent_scores(self, seconds=30):
        now    = time.time()
        cutoff = now - seconds
        pts    = [(t, s) for t, s in self.history if t >= cutoff]
        if not pts:
            return np.array([]), np.array([])
        times  = np.array([t - now for t, _ in pts])
        scores = np.array([s for _, s in pts])
        return times, scores

    def current_score(self):
        return self.history[-1][1] if self.history else 0.0

    def save_report(self, path="reports"):
        os.makedirs(path, exist_ok=True)
        ts    = time.strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(path, f"session_{ts}.json")
        scores = [s for _, s in self.history]
        data  = {
            "session_time":     ts,
            "question_results": self.question_scores,
            "score_history":    [(round(t, 2), round(s, 1)) for t, s in self.history],
            "summary": {
                "avg_score":       round(float(np.mean(scores)), 1) if scores else 0,
                "max_score":       round(float(np.max(scores)), 1)  if scores else 0,
                "total_questions": len(self.question_scores),
            },
        }
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        return fname
