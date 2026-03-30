"""
rppg/rppg_main_f1.py
5-Branch rPPG Heart-Rate Estimator
────────────────────────────────────
Branch 1 : POS + FFT  (signal processing)
Branch 2 : LSTM       (Keras / TensorFlow)
Branch 3 : RhythmFormer (3D Transformer, PyTorch)
Branch 4 : DeepPhys   (two-stream 2D CNN, PyTorch)  ← NEW
Branch 5 : PhysNet    (3D CNN, PyTorch)              ← NEW

Fusion    : 5-branch SNR-adaptive weighted average (fusion5.py)
Display   : live BPM HUD + rPPG waveform panel + stress indicator
"""

import sys
import os
import cv2
import time
import numpy as np
from collections import deque
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# ==============================
# PATH SETUP
# ==============================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from forehead_and_cheeks import get_rois
from buffer import SignalBuffer
from signal_processing import bandpass_filter, estimate_bpm
from model_predictor import LSTMPredictor
from rhythmformer_predictor import RhythmFormerPredictor
from deepphys_predictor import DeepPhysPredictor
from physnet_predictor import PhysNetPredictor
from fusion5 import dynamic_fusion_5branch

# ==============================
# CONFIG
# ==============================
WINDOW_SECONDS      = 15
FPS_INIT            = 30
MIN_PIXELS          = 50
BPM_UPDATE_INTERVAL = 1.0
MAX_BPM_JUMP        = 5
BPM_SMOOTHING       = 8
LOW_CUT             = 0.75
HIGH_CUT            = 3.0
RF_FRAMES           = 160

RHYTHMFORMER_WEIGHTS = os.path.join(PROJECT_ROOT, "UBFC-rPPG_RhythmFormer.pth")
DEEPPHYS_WEIGHTS     = os.path.join(PROJECT_ROOT, "deepphys.pth")   # optional
PHYSNET_WEIGHTS      = os.path.join(PROJECT_ROOT, "physnet.pth")    # optional

# ==============================
# SIGNAL BUFFERS (per ROI)
# ==============================
forehead_buffer    = SignalBuffer(WINDOW_SECONDS, FPS_INIT)
left_cheek_buffer  = SignalBuffer(WINDOW_SECONDS, FPS_INIT)
right_cheek_buffer = SignalBuffer(WINDOW_SECONDS, FPS_INIT)

# ==============================
# INIT MODELS
# ==============================
lstm = LSTMPredictor()
print("✅ LSTM loaded")

try:
    rf = RhythmFormerPredictor(RHYTHMFORMER_WEIGHTS, fps=FPS_INIT)
    rf_available = True
    print("✅ RhythmFormer loaded")
except Exception as e:
    print(f"⚠️  RhythmFormer not loaded: {e}")
    rf_available = False
    rf = None

try:
    dp = DeepPhysPredictor(weights_path=DEEPPHYS_WEIGHTS, fps=FPS_INIT)
    dp_available = True
    print("✅ DeepPhys loaded")
except Exception as e:
    print(f"⚠️  DeepPhys not loaded: {e}")
    dp_available = False
    dp = None

try:
    pn = PhysNetPredictor(weights_path=PHYSNET_WEIGHTS, fps=FPS_INIT)
    pn_available = True
    print("✅ PhysNet loaded")
except Exception as e:
    print(f"⚠️  PhysNet not loaded: {e}")
    pn_available = False
    pn = None

print("✅ All models initialised\n")

cap = cv2.VideoCapture(0)

# ==============================
# STATE
# ==============================
timestamps = []

bpm_sp_history    = deque(maxlen=6)
bpm_lstm_history  = deque(maxlen=6)
bpm_rf_history    = deque(maxlen=4)
bpm_dp_history    = deque(maxlen=4)
bpm_pn_history    = deque(maxlen=4)
bpm_final_history = deque(maxlen=BPM_SMOOTHING)

last_bpm      = None
last_bpm_time = 0

bpm_signal   = None
bpm_lstm     = None
bpm_rf       = None
bpm_chrom    = None
bpm_dp       = None
bpm_pn       = None
w_signal     = 0.0
w_lstm       = 0.0
w_rf         = 0.0
w_dp         = 0.0
w_pn         = 0.0
snr          = 0.0
confidence   = 0.0
stress       = 0
last_waveform = None

baseline_hrs  = []
baseline_hr   = None
BASELINE_SECS = 30


# ==============================
# SIGNAL PROCESSING FUNCTIONS
# ==============================

def pos_algorithm(rgb):
    """POS — Wang et al. 2017"""
    X  = rgb.T
    Xn = X / (np.mean(X, axis=1, keepdims=True) + 1e-8)
    S1 = Xn[1] - Xn[2]
    S2 = Xn[0] + Xn[1] - 2 * Xn[2]
    alpha = np.std(S1) / (np.std(S2) + 1e-8)
    return S1 - alpha * S2


def chrom_algorithm(rgb):
    """CHROM — De Haan & Jeanne 2013"""
    X  = rgb.T
    Xn = X / (np.mean(X, axis=1, keepdims=True) + 1e-8)
    Xs = 3 * Xn[0] - 2 * Xn[1]
    Ys = 1.5 * Xn[0] + Xn[1] - 1.5 * Xn[2]
    alpha = np.std(Xs) / (np.std(Ys) + 1e-8)
    return Xs - alpha * Ys


def fft_bpm(signal, fps, low=0.75, high=3.0):
    N     = len(signal)
    freqs = fftfreq(N, d=1.0 / fps)
    mag   = np.abs(fft(signal))
    valid = (freqs >= low) & (freqs <= high)
    if not valid.any():
        return None, 0.0
    v_f   = freqs[valid]
    v_m   = mag[valid]
    peak  = np.argmax(v_m)
    hr    = float(v_f[peak] * 60.0)
    snr_v = float(v_m[peak] ** 2 / (np.sum(v_m ** 2) + 1e-8))
    return hr, snr_v


def denoise_and_filter(signal, fps):
    from scipy.signal import detrend
    sig = detrend(np.array(signal, dtype=np.float64))
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    sig = bandpass_filter(sig, fps, low=LOW_CUT, high=HIGH_CUT)
    return sig


def harmonic_safe_bpm(bpm, prev_bpm=None):
    if bpm is None:
        return None
    if prev_bpm is not None:
        if abs(bpm - 2 * prev_bpm) < 10:
            bpm = bpm / 2.0
        elif abs(bpm * 2 - prev_bpm) < 10:
            bpm = bpm * 2.0
    if bpm < 45:
        bpm *= 2.0
    return float(np.clip(bpm, 42, 180))


def smooth_branch(new_val, history, max_jump=15):
    if new_val is None:
        return None, history
    if len(history) > 0:
        last = np.mean(list(history)[-3:])
        if abs(new_val - last) > max_jump:
            new_val = last + np.sign(new_val - last) * max_jump
    history.append(new_val)
    return float(np.mean(history)), history


def roi_to_rgb(frame, mask):
    if mask is None:
        return None
    pixels = frame[mask > 0]
    if pixels.shape[0] < MIN_PIXELS:
        return None
    return np.array([
        np.mean(pixels[:, 2]),
        np.mean(pixels[:, 1]),
        np.mean(pixels[:, 0])
    ])


def combine_roi_signals(fh_buf, lc_buf, rc_buf, fps):
    """Weighted multi-ROI combination: forehead 0.60, each cheek 0.20"""
    signals = []
    weights = []
    if fh_buf.ready():
        R, G, B = fh_buf.get()
        rgb = np.vstack([R, G, B]).T
        signals.append(denoise_and_filter(pos_algorithm(rgb), fps))
        weights.append(0.60)
    if lc_buf.ready():
        R, G, B = lc_buf.get()
        rgb = np.vstack([R, G, B]).T
        signals.append(denoise_and_filter(pos_algorithm(rgb), fps))
        weights.append(0.20)
    if rc_buf.ready():
        R, G, B = rc_buf.get()
        rgb = np.vstack([R, G, B]).T
        signals.append(denoise_and_filter(pos_algorithm(rgb), fps))
        weights.append(0.20)
    if not signals:
        return None
    min_len  = min(len(s) for s in signals)
    signals  = [s[-min_len:] for s in signals]
    total_w  = sum(weights[:len(signals)])
    return sum(w / total_w * s for w, s in zip(weights, signals))


def compute_confidence(snr_val, *estimates):
    snr_score = float(np.clip(snr_val / 0.25, 0, 1))
    valid = [b for b in estimates if b is not None]
    if len(valid) >= 2:
        spread    = max(valid) - min(valid)
        agreement = float(np.clip(1.0 - spread / 20.0, 0, 1))
    else:
        agreement = 0.4
    return round(0.5 * snr_score + 0.5 * agreement, 3)


def compute_stress(bpm_val, baseline):
    if baseline is None or bpm_val is None:
        return 0
    elevation = bpm_val - baseline
    return int(np.clip((elevation / 30.0) * 100, 0, 100))


def draw_rois(frame, rois):
    if rois["forehead"] is None:
        return frame
    overlay = frame.copy()
    overlay[rois["forehead"]    > 0] = [180, 100,  0]
    overlay[rois["left_cheek"]  > 0] = [  0, 180, 80]
    overlay[rois["right_cheek"] > 0] = [  0, 180, 80]
    return cv2.addWeighted(frame, 0.72, overlay, 0.28, 0)


def draw_waveform(frame, signal, label="rPPG Waveform"):
    """Draw a live rPPG waveform panel in the bottom-right corner."""
    if signal is None or len(signal) < 2:
        return frame
    fh, fw = frame.shape[:2]
    pw, ph  = 280, 90
    margin  = 12
    x0 = fw - pw - margin
    y0 = fh - ph - margin
    x1, y1 = x0 + pw, y0 + ph

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 1)
    cv2.putText(frame, label, (x0 + 6, y0 + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    sig = np.array(signal, dtype=np.float32)
    sig = sig - sig.mean()
    peak = np.abs(sig).max()
    if peak < 1e-6:
        return frame
    sig = sig / peak
    n_pts = pw - 12
    if len(sig) > n_pts:
        sig = sig[-n_pts:]

    mid_y  = y0 + ph // 2 + 8
    amp    = (ph // 2) - 14
    x_step = (pw - 12) / max(len(sig) - 1, 1)
    pts = [(int(x0 + 6 + i * x_step), int(mid_y - v * amp))
           for i, v in enumerate(sig)]
    cv2.line(frame, (x0 + 6, mid_y), (x1 - 6, mid_y), (60, 60, 60), 1)
    for i in range(1, len(pts)):
        colour = (0, 220, 120) if sig[i] > 0 else (0, 180, 255)
        cv2.line(frame, pts[i-1], pts[i], colour, 1)
    return frame


# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rois = get_rois(frame)

    now = time.time()
    timestamps.append(now)
    timestamps = timestamps[-30:]
    fs = (len(timestamps) / (timestamps[-1] - timestamps[0] + 1e-6)
          if len(timestamps) >= 2 else FPS_INIT)

    # RGB extraction per ROI
    if rois["valid"]:
        fh_rgb = roi_to_rgb(frame, rois["forehead"])
        if fh_rgb is not None:
            forehead_buffer.append(*fh_rgb)

        lc_rgb = roi_to_rgb(frame, rois["left_cheek"])
        if lc_rgb is not None:
            left_cheek_buffer.append(*lc_rgb)

        rc_rgb = roi_to_rgb(frame, rois["right_cheek"])
        if rc_rgb is not None:
            right_cheek_buffer.append(*rc_rgb)

        if rf_available and rf is not None:
            rf.add_frame(frame)
        if dp_available and dp is not None:
            dp.add_frame(frame)
        if pn_available and pn is not None:
            pn.add_frame(frame)

    # BPM computation
    if forehead_buffer.ready() and fs > 5:
        if time.time() - last_bpm_time >= BPM_UPDATE_INTERVAL:

            combined_sig = combine_roi_signals(
                forehead_buffer, left_cheek_buffer,
                right_cheek_buffer, fs
            )

            if combined_sig is not None:
                last_waveform = combined_sig

                # Branch 1A: POS + FFT
                raw_sp, snr = fft_bpm(combined_sig, fs)
                raw_sp = harmonic_safe_bpm(raw_sp, last_bpm)
                bpm_signal, bpm_sp_history = smooth_branch(
                    raw_sp, bpm_sp_history, max_jump=12)

                # Branch 1B: CHROM + FFT (display only)
                if forehead_buffer.ready():
                    R, G, B = forehead_buffer.get()
                    chrom_s = chrom_algorithm(np.vstack([R, G, B]).T)
                    chrom_f = denoise_and_filter(chrom_s, fs)
                    raw_chrom, _ = fft_bpm(chrom_f, fs)
                    bpm_chrom = harmonic_safe_bpm(raw_chrom, last_bpm)

                # Branch 2: LSTM
                try:
                    raw_lstm = lstm.predict(combined_sig)
                    raw_lstm = harmonic_safe_bpm(raw_lstm, last_bpm)
                    bpm_lstm, bpm_lstm_history = smooth_branch(
                        raw_lstm, bpm_lstm_history, max_jump=12)
                except Exception:
                    bpm_lstm = bpm_signal

                # Branch 3: RhythmFormer
                bpm_rf = None
                if rf_available and rf is not None and rf.ready():
                    try:
                        raw_rf = rf.predict()
                        raw_rf = harmonic_safe_bpm(raw_rf, last_bpm)
                        bpm_rf, bpm_rf_history = smooth_branch(
                            raw_rf, bpm_rf_history, max_jump=10)
                    except Exception:
                        bpm_rf = None

                # Branch 4: DeepPhys
                bpm_dp = None
                if dp_available and dp is not None and dp.ready():
                    try:
                        raw_dp = dp.predict()
                        raw_dp = harmonic_safe_bpm(raw_dp, last_bpm)
                        bpm_dp, bpm_dp_history = smooth_branch(
                            raw_dp, bpm_dp_history, max_jump=10)
                    except Exception:
                        bpm_dp = None

                # Branch 5: PhysNet
                bpm_pn = None
                if pn_available and pn is not None and pn.ready():
                    try:
                        raw_pn = pn.predict()
                        raw_pn = harmonic_safe_bpm(raw_pn, last_bpm)
                        bpm_pn, bpm_pn_history = smooth_branch(
                            raw_pn, bpm_pn_history, max_jump=10)
                    except Exception:
                        bpm_pn = None

                # 5-Branch Dynamic Fusion
                bpm_fused, w_signal, w_lstm, w_rf, w_dp, w_pn, snr = \
                    dynamic_fusion_5branch(
                        bpm_signal or 75.0,
                        bpm_lstm   or 75.0,
                        bpm_rf,
                        bpm_dp,
                        bpm_pn,
                        combined_sig,
                        fps=fs
                    )

                # Physiological constraint
                if last_bpm is not None:
                    bpm_fused = float(np.clip(
                        bpm_fused,
                        last_bpm - MAX_BPM_JUMP,
                        last_bpm + MAX_BPM_JUMP
                    ))

                bpm_final_history.append(bpm_fused)
                last_bpm      = float(np.mean(bpm_final_history))
                last_bpm_time = time.time()

                confidence = compute_confidence(
                    snr, bpm_signal, bpm_lstm, bpm_rf, bpm_dp, bpm_pn)

                if baseline_hr is None:
                    baseline_hrs.append(last_bpm)
                    if len(baseline_hrs) >= BASELINE_SECS:
                        baseline_hr = float(np.mean(baseline_hrs))
                        print(f"📊 Baseline HR: {baseline_hr:.1f} BPM")

                stress = compute_stress(last_bpm, baseline_hr)

    # ── Overlays ──────────────────────────────────────────────────────────────
    frame = draw_rois(frame, rois)
    frame = draw_waveform(frame, last_waveform)

    # HUD
    y = 35
    if last_bpm is not None:
        cv2.putText(frame, f"BPM: {int(last_bpm)}",
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
        y += 45

        cv2.putText(frame, "--- Branches ---",
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (160, 160, 160), 1)
        y += 20

        if bpm_signal is not None:
            cv2.putText(frame,
                        f"POS+FFT  : {int(bpm_signal):3d} BPM  w={w_signal:.2f}",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 0), 2)
            y += 22

        if bpm_chrom is not None:
            cv2.putText(frame,
                        f"CHROM+FFT: {int(bpm_chrom):3d} BPM",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (200, 220, 0), 1)
            y += 20

        if bpm_lstm is not None:
            cv2.putText(frame,
                        f"LSTM     : {int(bpm_lstm):3d} BPM  w={w_lstm:.2f}",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 165, 0), 2)
            y += 22

        warmup_rf = len(rf.frame_buffer) if rf is not None else 0
        if bpm_rf is not None:
            cv2.putText(frame,
                        f"RhythmF  : {int(bpm_rf):3d} BPM  w={w_rf:.2f}",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (150, 255, 150), 2)
        else:
            cv2.putText(frame,
                        f"RhythmF  : warming ({warmup_rf}/{RF_FRAMES})",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (120, 120, 120), 1)
        y += 22

        warmup_dp = len(dp.frame_buffer) if dp is not None else 0
        if bpm_dp is not None:
            cv2.putText(frame,
                        f"DeepPhys : {int(bpm_dp):3d} BPM  w={w_dp:.2f}",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 150, 255), 2)
        else:
            cv2.putText(frame,
                        f"DeepPhys : warming ({warmup_dp}/150)",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (120, 120, 120), 1)
        y += 22

        warmup_pn = len(pn.frame_buffer) if pn is not None else 0
        if bpm_pn is not None:
            cv2.putText(frame,
                        f"PhysNet  : {int(bpm_pn):3d} BPM  w={w_pn:.2f}",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 100, 200), 2)
        else:
            cv2.putText(frame,
                        f"PhysNet  : warming ({warmup_pn}/32)",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (120, 120, 120), 1)
        y += 25

        cv2.line(frame, (30, y), (460, y), (70, 70, 70), 1)
        y += 14

        snr_color = (
            (0, 255, 0) if snr > 0.25 else
            (0, 165, 255) if snr > 0.10 else
            (0, 0, 255)
        )
        cv2.putText(frame, f"SNR      : {snr:.3f}",
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, snr_color, 1)
        y += 22

        conf_color = (
            (0, 255, 0) if confidence >= 0.65 else
            (0, 165, 255) if confidence >= 0.40 else
            (0, 0, 255)
        )
        conf_label = (
            "HIGH" if confidence >= 0.65 else
            "MEDIUM" if confidence >= 0.40 else "LOW"
        )
        cv2.putText(frame,
                    f"Conf     : {confidence:.2f} [{conf_label}]",
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, conf_color, 2)
        y += 22

        if baseline_hr is not None:
            stress_color = (
                (0, 255, 0) if stress < 35 else
                (0, 165, 255) if stress < 65 else
                (0, 0, 255)
            )
            stress_label = (
                "LOW" if stress < 35 else
                "MODERATE" if stress < 65 else "HIGH"
            )
            cv2.putText(frame,
                        f"Stress   : {stress:3d}/100 [{stress_label}]",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, stress_color, 2)
        else:
            remaining = max(0, BASELINE_SECS - len(baseline_hrs))
            cv2.putText(frame,
                        f"Stress   : calibrating ({remaining}s)",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (120, 120, 120), 1)
        y += 22

        if rois.get("glasses_detected"):
            cv2.putText(frame,
                        "Glasses detected - reduced accuracy",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 165, 255), 1)

    if not rois["valid"]:
        for idx, alert in enumerate(rois.get("alerts", ["Face not detected"])):
            cv2.putText(frame, alert,
                        (30, 220 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

    if not forehead_buffer.ready():
        fh_dim, fw_dim = frame.shape[:2]
        pct = min(200, int(
            (len(timestamps) / (WINDOW_SECONDS * FPS_INIT)) * 200))
        cv2.rectangle(frame, (30, fh_dim-30), (230, fh_dim-15),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (30, fh_dim-30), (30+pct, fh_dim-15),
                      (0, 200, 100), -1)
        cv2.putText(frame, "Warming up...",
                    (238, fh_dim-17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

    cv2.imshow("rPPG — 5 Branch Fusion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")
