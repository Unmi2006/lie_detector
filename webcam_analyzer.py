"""
Webcam Analyzer Module
Analyzes: blink rate, eye/iris movement, head pose, facial tension
"""

import cv2
import mediapipe as mp
import numpy as np
import collections
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

# ─── Landmark indices ────────────────────────────────────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP   = 1
# Eyebrow raise: inner brow vs eye distance
LEFT_BROW  = [336, 296, 334, 293, 300]
RIGHT_BROW = [107,  66,  105, 63,  70]

EAR_THRESHOLD  = 0.20
BLINK_CONSEC   = 2
WINDOW_SECONDS = 10
FPS            = 30


def _ear(landmarks, indices, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _iris_offset(landmarks, iris_idx, eye_idx, w, h):
    iris = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in iris_idx])
    eye  = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_idx])
    cx   = iris[:, 0].mean()
    el, er = eye[:, 0].min(), eye[:, 0].max()
    return (cx - el) / (er - el + 1e-6) * 2 - 1


class WebcamAnalyzer:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.cap = None

        # State
        self.eye_closed_frames = 0
        self.blink_timestamps  = collections.deque()
        self.eye_move_history  = collections.deque(maxlen=FPS * WINDOW_SECONDS)
        self.head_move_history = collections.deque(maxlen=FPS * WINDOW_SECONDS)

        self.baseline_nose     = None
        self.baseline_blinks   = 15.0
        self.baseline_eye_std  = 0.05
        self.baseline_head     = 2.0
        self.calibrated        = False

        self.blink_total    = 0
        self.start_time     = time.time()

        # Live metrics
        self.blinks_per_min = 0.0
        self.eye_std        = 0.0
        self.head_avg       = 0.0
        self.ear_value      = 0.3
        self.face_detected  = False
        self.frame_rgb      = None   # latest processed frame (BGR)

    def open_camera(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.start_time = time.time()
        return self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()
        self.face_mesh.close()

    def calibrate(self):
        self.baseline_blinks  = max(self.blink_total * (60 / max(time.time() - self.start_time, 1)), 5)
        self.baseline_eye_std = float(np.std(list(self.eye_move_history))) if self.eye_move_history else 0.05
        self.baseline_head    = float(np.mean(list(self.head_move_history))) if self.head_move_history else 2.0
        self.calibrated       = True
        self.blink_total      = 0
        self.blink_timestamps.clear()
        self.start_time       = time.time()

    def process_frame(self):
        """Read + process one frame. Returns annotated BGR frame or None."""
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        now   = time.time()

        self.face_detected = bool(results.multi_face_landmarks)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # EAR
            ear = (_ear(lm, LEFT_EYE, w, h) + _ear(lm, RIGHT_EYE, w, h)) / 2.0
            self.ear_value = ear

            if ear < EAR_THRESHOLD:
                self.eye_closed_frames += 1
            else:
                if self.eye_closed_frames >= BLINK_CONSEC:
                    self.blink_total += 1
                    self.blink_timestamps.append(now)
                self.eye_closed_frames = 0

            # Iris
            il = _iris_offset(lm, LEFT_IRIS,  LEFT_EYE,  w, h)
            ir = _iris_offset(lm, RIGHT_IRIS, RIGHT_EYE, w, h)
            self.eye_move_history.append((il + ir) / 2.0)

            # Head
            nx, ny = lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h
            if self.baseline_nose is None:
                self.baseline_nose = (nx, ny)
            dist = np.linalg.norm(np.array([nx, ny]) - np.array(self.baseline_nose))
            self.head_move_history.append(dist)

            # Clean old blink timestamps
            cutoff = now - WINDOW_SECONDS
            while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
                self.blink_timestamps.popleft()

            # Rolling metrics
            win = max(now - self.start_time, 1)
            self.blinks_per_min = (len(self.blink_timestamps) / min(win, WINDOW_SECONDS)) * 60
            self.eye_std  = float(np.std(list(self.eye_move_history)))
            self.head_avg = float(np.mean(list(self.head_move_history)))

            # Draw subtle mesh
            mp_drawing.draw_landmarks(
                frame, results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(40, 60, 80), thickness=1, circle_radius=0
                )
            )

        self.frame_rgb = frame
        return frame

    def get_visual_score(self):
        if not self.calibrated:
            return 0.0
        score = 0.0
        blink_ratio = self.blinks_per_min / max(self.baseline_blinks, 1)
        score += float(np.clip((blink_ratio - 1.0) * 30, 0, 35))
        eye_ratio = self.eye_std / max(self.baseline_eye_std, 0.01)
        score += float(np.clip((eye_ratio - 1.0) * 25, 0, 35))
        head_ratio = self.head_avg / max(self.baseline_head, 0.5)
        score += float(np.clip((head_ratio - 1.0) * 15, 0, 30))
        return min(score, 100.0)

    def snapshot(self):
        return {
            "blinks_per_min": round(self.blinks_per_min, 1),
            "eye_std":        round(self.eye_std, 4),
            "head_avg":       round(self.head_avg, 2),
            "ear":            round(self.ear_value, 3),
            "visual_score":   round(self.get_visual_score(), 1),
        }
