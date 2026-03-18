"""
Webcam Analyzer — MediaPipe-FREE version
Uses only OpenCV (Haar cascades + dlib-style geometry)
No MediaPipe, no protobuf conflicts!
Detects: blink rate, eye movement, head movement
"""

import cv2
import numpy as np
import collections
import time

# ─── Load OpenCV built-in cascades ───────────────────────────────────────────
_BASE = cv2.data.haarcascades
FACE_CASCADE = cv2.CascadeClassifier(_BASE + "haarcascade_frontalface_default.xml")
EYE_CASCADE  = cv2.CascadeClassifier(_BASE + "haarcascade_eye.xml")

WINDOW_SECONDS = 10
FPS            = 30
EAR_THRESHOLD  = 0.22
BLINK_CONSEC   = 2


def _eye_aspect_ratio(eye_rect, gray):
    """Estimate EAR from eye bounding box pixel intensity."""
    x, y, w, h = eye_rect
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.3
    # Use vertical vs horizontal ratio of bright region
    col_means = roi.mean(axis=0)
    row_means = roi.mean(axis=1)
    v = row_means.std() + 1e-6
    hz = col_means.std() + 1e-6
    ear = min(v / (hz + v), 0.5)
    return float(ear)


class WebcamAnalyzer:
    def __init__(self):
        self.cap = None

        # Blink state
        self.eye_closed_frames  = 0
        self.blink_timestamps   = collections.deque()
        self.blink_total        = 0

        # Eye/head history
        self.eye_move_history  = collections.deque(maxlen=FPS * WINDOW_SECONDS)
        self.head_move_history = collections.deque(maxlen=FPS * WINDOW_SECONDS)

        # Baseline
        self.baseline_nose     = None
        self.baseline_blinks   = 15.0
        self.baseline_eye_std  = 0.05
        self.baseline_head     = 2.0
        self.calibrated        = False

        self.start_time        = time.time()

        # Live metrics
        self.blinks_per_min    = 0.0
        self.eye_std           = 0.0
        self.head_avg          = 0.0
        self.ear_value         = 0.3
        self.face_detected     = False

        # Dummy face_mesh attribute so rPPG code doesn't break
        self.face_mesh         = _DummyFaceMesh()

    def open_camera(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.start_time = time.time()
        return self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()

    def calibrate(self):
        self.baseline_blinks  = max(
            self.blink_total * (60 / max(time.time()-self.start_time, 1)), 5)
        self.baseline_eye_std = float(np.std(list(self.eye_move_history))) \
                                if self.eye_move_history else 0.05
        self.baseline_head    = float(np.mean(list(self.head_move_history))) \
                                if self.head_move_history else 2.0
        self.calibrated       = True
        self.blink_total      = 0
        self.blink_timestamps.clear()
        self.start_time       = time.time()

    def process_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w  = frame.shape[:2]
        now   = time.time()

        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)

        self.face_detected = len(faces) > 0

        if len(faces) > 0:
            # Largest face
            fx, fy, fw, fh = max(faces, key=lambda r: r[2]*r[3])

            # Draw face box
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (40, 180, 160), 1)

            # Head center as proxy for movement
            cx, cy = fx + fw//2, fy + fh//2
            if self.baseline_nose is None:
                self.baseline_nose = (cx, cy)
            dist = float(np.linalg.norm(
                np.array([cx, cy]) - np.array(self.baseline_nose)))
            self.head_move_history.append(dist)

            # Eye detection inside face ROI
            face_roi_gray  = gray[fy:fy+fh, fx:fx+fw]
            face_roi_color = frame[fy:fy+fh, fx:fx+fw]
            eyes = EYE_CASCADE.detectMultiScale(
                face_roi_gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(20, 20))

            eye_centers_x = []
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(face_roi_color,
                              (ex, ey), (ex+ew, ey+eh), (0, 200, 100), 1)
                eye_centers_x.append(ex + ew//2)

                # EAR estimate
                ear = _eye_aspect_ratio(
                    (fx+ex, fy+ey, ew, eh), gray)
                self.ear_value = ear

            # Blink detection
            if len(eyes) == 0:
                self.eye_closed_frames += 1
            else:
                if self.eye_closed_frames >= BLINK_CONSEC:
                    self.blink_total += 1
                    self.blink_timestamps.append(now)
                self.eye_closed_frames = 0

            # Eye horizontal position
            if eye_centers_x:
                norm_x = np.mean(eye_centers_x) / max(fw, 1) * 2 - 1
                self.eye_move_history.append(float(norm_x))

            # Clean old blinks
            cutoff = now - WINDOW_SECONDS
            while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
                self.blink_timestamps.popleft()

            # Update metrics
            win = max(now - self.start_time, 1)
            self.blinks_per_min = (len(self.blink_timestamps) /
                                   min(win, WINDOW_SECONDS)) * 60
            self.eye_std  = float(np.std(list(self.eye_move_history))) \
                            if self.eye_move_history else 0.0
            self.head_avg = float(np.mean(list(self.head_move_history))) \
                            if self.head_move_history else 0.0

            # HUD overlay
            label = f"BLINKS: {self.blinks_per_min:.0f}/min"
            cv2.putText(frame, label, (10, h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,200,200), 1)

        return frame

    def get_visual_score(self):
        if not self.calibrated:
            return 0.0
        score = 0.0
        blink_ratio = self.blinks_per_min / max(self.baseline_blinks, 1)
        score += float(np.clip((blink_ratio - 1.0) * 30, 0, 35))
        eye_ratio   = self.eye_std / max(self.baseline_eye_std, 0.01)
        score += float(np.clip((eye_ratio - 1.0) * 25, 0, 35))
        head_ratio  = self.head_avg / max(self.baseline_head, 0.5)
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


class _DummyFaceMesh:
    """Stub so rPPG analyzer doesn't crash when accessing face_mesh."""
    def process(self, frame):
        return _DummyResult()


class _DummyResult:
    multi_face_landmarks = None
