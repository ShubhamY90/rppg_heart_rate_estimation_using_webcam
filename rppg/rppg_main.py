import sys
import os
import cv2
import time
import numpy as np
from collections import deque

# ==============================
# FIX IMPORT PATH
# ==============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from forehead_and_cheeks import get_rois
from buffer import SignalBuffer
from signal_processing import bandpass_filter, estimate_bpm

# ==============================
# CONFIG (IMPROVED)
# ==============================
WINDOW_SECONDS = 15          # Increased for better frequency resolution
FPS_INIT = 30
MIN_PIXELS = 50

BPM_UPDATE_INTERVAL = 1.0
MAX_BPM_JUMP = 8
BPM_SMOOTHING = 5

LOW_CUT = 0.7                # Expanded slightly
HIGH_CUT = 3.5

# ==============================
# INIT
# ==============================
cap = cv2.VideoCapture(0)
buffer = SignalBuffer(WINDOW_SECONDS, FPS_INIT)

timestamps = []
bpm_history = deque(maxlen=BPM_SMOOTHING)

last_bpm = None
last_bpm_time = 0

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rois = get_rois(frame)

    # ==============================
    # FPS ESTIMATION
    # ==============================
    now = time.time()
    timestamps.append(now)
    timestamps = timestamps[-30:]

    if len(timestamps) >= 2:
        fs = len(timestamps) / (timestamps[-1] - timestamps[0] + 1e-6)
    else:
        fs = FPS_INIT

    # ==============================
    # RGB EXTRACTION
    # ==============================
    if rois["valid"] and rois["forehead"] is not None:
        mask = rois["forehead"]
        pixels = frame[mask > 0]

        if pixels.shape[0] > MIN_PIXELS:
            B, G, R = np.mean(pixels, axis=0)
            buffer.append(R, G, B)

    # ==============================
    # BPM COMPUTATION
    # ==============================
    if buffer.ready() and fs > 5:
        if time.time() - last_bpm_time >= BPM_UPDATE_INTERVAL:

            R, G, B = buffer.get()

            # ------------------------------
            # 1. Normalize RGB (mean-based)
            # ------------------------------
            rgb = np.vstack([R, G, B]).T
            rgb = rgb / (np.mean(rgb, axis=0) + 1e-6)

            # ------------------------------
            # 2. POS Algorithm
            # ------------------------------
            X = 3 * rgb[:, 0] - 2 * rgb[:, 1]
            Y = 1.5 * rgb[:, 0] + rgb[:, 1] - 1.5 * rgb[:, 2]

            alpha = np.std(X) / (np.std(Y) + 1e-6)
            pos_signal = X - alpha * Y

            # ------------------------------
            # 3. Remove DC + Normalize
            # ------------------------------
            pos_signal -= np.mean(pos_signal)
            pos_signal /= (np.std(pos_signal) + 1e-6)

            # ------------------------------
            # 4. Bandpass Filter
            # ------------------------------
            filtered = bandpass_filter(
                pos_signal,
                fs,
                low=LOW_CUT,
                high=HIGH_CUT
            )

            # ------------------------------
            # 5. Estimate BPM
            # ------------------------------
            bpm = estimate_bpm(filtered, fs)

            # ------------------------------
            # 6. Harmonic Correction
            # ------------------------------
            if bpm < 55:
                bpm *= 2

            # ------------------------------
            # 7. Physiological Constraint
            # ------------------------------
            if last_bpm is not None:
                bpm = np.clip(
                    bpm,
                    last_bpm - MAX_BPM_JUMP,
                    last_bpm + MAX_BPM_JUMP
                )

            bpm_history.append(bpm)
            last_bpm = np.mean(bpm_history)
            last_bpm_time = time.time()

    # ==============================
    # DISPLAY
    # ==============================
    if last_bpm is not None:
        cv2.putText(
            frame,
            f"BPM: {int(last_bpm)}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

    if not rois["valid"]:
        cv2.putText(
            frame,
            "Adjust face / lighting",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("rPPG", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()