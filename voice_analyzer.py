"""
Voice Analyzer Module
Analyzes: pitch variation, speech rate, pause duration, voice tremor, filler words
"""

import pyaudio
import numpy as np
import librosa
import collections
import time
import threading
import speech_recognition as sr

# ─── Config ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHUNK_SIZE     = 1024
CHANNELS       = 1
FORMAT         = pyaudio.paInt16
BUFFER_SECONDS = 3       # Rolling analysis window
SILENCE_THRESH = 500     # RMS below this = silence

FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally",
                "actually", "i mean", "right", "so"}


class VoiceAnalyzer:
    def __init__(self):
        self.pa        = pyaudio.PyAudio()
        self.stream    = None
        self.running   = False
        self.lock      = threading.Lock()

        # Rolling audio buffer
        buf_size = SAMPLE_RATE * BUFFER_SECONDS
        self.audio_buffer = collections.deque(maxlen=buf_size)

        # Metrics (updated continuously)
        self.pitch_mean      = 0.0
        self.pitch_std       = 0.0
        self.tremor_score    = 0.0
        self.speech_rate     = 0.0   # estimated syllables/sec
        self.pause_ratio     = 0.0   # fraction of time silent
        self.filler_count    = 0
        self.rms_level       = 0.0
        self.waveform        = np.zeros(512)

        # Baseline (set during calibration)
        self.baseline_pitch_mean = 180.0
        self.baseline_pitch_std  = 15.0
        self.baseline_tremor     = 0.02
        self.baseline_pause      = 0.2
        self.calibrated          = False

        # Async speech recognition
        self.recognizer     = sr.Recognizer()
        self.filler_thread  = None
        self.last_transcribe = time.time()

    # ── Stream control ────────────────────────────────────────────────────────
    def start(self):
        self.stream = self.pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._audio_callback
        )
        self.running = True
        self.stream.start_stream()
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()

    # ── PyAudio callback ─────────────────────────────────────────────────────
    def _audio_callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        with self.lock:
            self.audio_buffer.extend(samples)
        return (in_data, pyaudio.paContinue)

    # ── Analysis loop ─────────────────────────────────────────────────────────
    def _analysis_loop(self):
        while self.running:
            time.sleep(0.15)
            with self.lock:
                if len(self.audio_buffer) < SAMPLE_RATE:
                    continue
                audio = np.array(self.audio_buffer, dtype=np.float32)

            audio_norm = audio / 32768.0

            # RMS
            self.rms_level = float(np.sqrt(np.mean(audio_norm ** 2)))

            # Waveform for display (downsample to 512 points)
            step = max(1, len(audio_norm) // 512)
            self.waveform = audio_norm[::step][:512]

            # Pitch analysis via librosa YIN
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    audio_norm,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=SAMPLE_RATE
                )
                voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
                if len(voiced) > 5:
                    self.pitch_mean = float(np.median(voiced))
                    self.pitch_std  = float(np.std(voiced))
                    # Tremor: std of pitch in short frames
                    frame_stds = [np.std(voiced[i:i+10]) for i in range(0, len(voiced)-10, 5)]
                    self.tremor_score = float(np.mean(frame_stds)) if frame_stds else 0.0
                else:
                    self.pitch_mean  = 0.0
                    self.pitch_std   = 0.0
                    self.tremor_score = 0.0
            except Exception:
                pass

            # Pause ratio (fraction of frames below silence threshold)
            rms_frames = librosa.feature.rms(y=audio_norm, frame_length=512, hop_length=256)[0]
            silence_mask = rms_frames < (SILENCE_THRESH / 32768.0)
            self.pause_ratio = float(silence_mask.mean())

            # Speech rate estimate: zero-crossing rate as syllable proxy
            zcr = librosa.feature.zero_crossing_rate(audio_norm)[0]
            self.speech_rate = float(zcr.mean() * SAMPLE_RATE / 512)

    # ── Calibration ──────────────────────────────────────────────────────────
    def calibrate(self):
        """Call after ~5s of speaking normally to set baseline."""
        self.baseline_pitch_mean = self.pitch_mean if self.pitch_mean > 0 else 180.0
        self.baseline_pitch_std  = self.pitch_std  if self.pitch_std  > 0 else 15.0
        self.baseline_tremor     = self.tremor_score if self.tremor_score > 0 else 0.02
        self.baseline_pause      = self.pause_ratio
        self.calibrated          = True

    # ── Async filler word detection ──────────────────────────────────────────
    def transcribe_async(self):
        """Periodically run STT to count filler words."""
        if time.time() - self.last_transcribe < 5.0:
            return
        self.last_transcribe = time.time()

        with self.lock:
            if len(self.audio_buffer) < SAMPLE_RATE * 2:
                return
            chunk = np.array(list(self.audio_buffer)[-SAMPLE_RATE*3:], dtype=np.int16)

        def _run():
            try:
                audio_data = sr.AudioData(chunk.tobytes(), SAMPLE_RATE, 2)
                text = self.recognizer.recognize_google(audio_data).lower()
                count = sum(1 for w in FILLER_WORDS if w in text)
                self.filler_count += count
            except Exception:
                pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── Voice lie score ──────────────────────────────────────────────────────
    def get_voice_score(self):
        """Returns 0-100 voice stress score."""
        if not self.calibrated or self.pitch_mean == 0:
            return 0.0

        score = 0.0

        # Pitch elevation vs baseline
        pitch_ratio = self.pitch_mean / max(self.baseline_pitch_mean, 1)
        score += np.clip((pitch_ratio - 1.0) * 40, 0, 30)

        # Pitch instability
        std_ratio = self.pitch_std / max(self.baseline_pitch_std, 1)
        score += np.clip((std_ratio - 1.0) * 20, 0, 25)

        # Tremor
        tremor_ratio = self.tremor_score / max(self.baseline_tremor, 0.001)
        score += np.clip((tremor_ratio - 1.0) * 15, 0, 25)

        # Pause ratio change
        pause_delta = abs(self.pause_ratio - self.baseline_pause)
        score += np.clip(pause_delta * 60, 0, 20)

        return float(min(score, 100))

    # ── Snapshot for report ──────────────────────────────────────────────────
    def snapshot(self):
        return {
            "pitch_mean":   round(self.pitch_mean, 1),
            "pitch_std":    round(self.pitch_std, 1),
            "tremor":       round(self.tremor_score, 4),
            "pause_ratio":  round(self.pause_ratio, 3),
            "speech_rate":  round(self.speech_rate, 3),
            "filler_count": self.filler_count,
            "rms":          round(self.rms_level, 4),
            "voice_score":  round(self.get_voice_score(), 1),
        }
