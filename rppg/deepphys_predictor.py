# rppg/deepphys_predictor.py
# DeepPhys: Learning Spatio-Temporal Features for Non-Contact Physiological
# Measurement (Liu et al., 2020) — PyTorch implementation.
#
# Architecture: Two-stream 2D CNN
#   Appearance stream  : processes the raw RGB frame
#   Motion stream      : processes the normalised temporal difference frame
#   Both streams share attention maps that guide feature extraction.
#
# The model outputs one BVP value per frame; we buffer the values and
# convert the waveform to BPM via FFT, matching the interface of the
# other predictors in this project.
#
# Weights: pass weights_path=None to run with random initialisation
# (useful for integration testing before real weights are available).

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq


# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────

class AppearanceStream(nn.Module):
    """Process raw RGB frame → attention map A."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(32,  1, 1),
            nn.Sigmoid()   # attention map in [0, 1]
        )

    def forward(self, x):       # x: (B, 3, H, W)
        return self.conv(x)     # (B, 1, H/4, W/4)


class MotionStream(nn.Module):
    """Process normalised diff frame → features gated by attention."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, diff, attn):   # diff: (B, 3, H, W), attn: (B, 1, H/4, W/4)
        feat = self.conv(diff)        # (B, 32, H/4, W/4)
        feat = feat * attn            # spatial attention gating
        return self.final(feat)       # (B, 1, H/4, W/4)


class DeepPhys(nn.Module):
    """
    Full DeepPhys network.
    Input : two tensors  (B, 3, H, W): raw frame + normalised diff frame
    Output: (B,) — one BVP scalar per sample in the batch
    """
    def __init__(self):
        super().__init__()
        self.appear = AppearanceStream()
        self.motion = MotionStream()
        # 2 maps (appear + motion) → scalar
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),   # → 2×(B,1,4,4)
            nn.Flatten(),              # 2 × 16 = 32
        )
        self.head = nn.Linear(32, 1)

    def forward(self, raw, diff):
        attn  = self.appear(raw)
        feat  = self.motion(diff, attn)
        # concatenate appearance attention + motion feature spatially
        attn_ds = attn                             # already H/4, W/4
        combined = torch.cat([attn_ds, feat], 1)  # (B, 2, H/4, W/4)
        flat = nn.functional.adaptive_avg_pool2d(combined, 4)
        flat = flat.reshape(flat.size(0), -1)      # (B, 2*16)
        return self.head(flat).squeeze(1)          # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# Predictor wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DeepPhysPredictor:
    """
    Maintains a frame buffer, feeds consecutive (frame, diff_frame) pairs
    to DeepPhys, accumulates the per-frame BVP output, then estimates BPM
    via FFT — identical interface to RhythmFormerPredictor.

    Usage:
        dp = DeepPhysPredictor(weights_path='deepphys.pth')  # or None
        dp.add_frame(bgr_frame)
        bpm = dp.predict()   # float or None
    """

    T = 150    # minimum frames before we produce a BPM estimate
    H = 36     # spatial resize (DeepPhys was trained at 36×36)
    W = 36

    def __init__(self, weights_path=None, fps=30):
        self.fps          = fps
        self.frame_buffer = deque(maxlen=self.T)   # raw frames (np float32 RGB)
        self.bvp_buffer   = deque(maxlen=self.T)   # accumulated BVP scalars
        self.device       = 'cpu'

        self.model = DeepPhys()
        self.model.eval()

        if weights_path and os.path.isfile(weights_path):
            try:
                state = torch.load(weights_path, map_location='cpu')
                # strip DataParallel prefix if present
                state = {k.replace('module.', ''): v for k, v in state.items()}
                self.model.load_state_dict(state, strict=False)
                print(f"✅ DeepPhys weights loaded from {weights_path}")
            except Exception as e:
                print(f"⚠️  DeepPhys weight loading failed: {e}  (using random init)")
        else:
            print("⚠️  DeepPhys: no weights found — running with random init")

    def add_frame(self, bgr_frame):
        """Add one BGR webcam frame."""
        resized = cv2.resize(bgr_frame, (self.W, self.H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.frame_buffer.append(rgb)

        # Compute BVP for the NEW frame if we have a previous frame
        if len(self.frame_buffer) >= 2:
            self._infer_one()

    def _infer_one(self):
        """Run one forward pass on the last two frames."""
        frames = list(self.frame_buffer)
        cur  = frames[-1]   # (H, W, 3)
        prev = frames[-2]

        # Normalised difference (motion) frame
        diff = (cur - prev)
        dnorm = diff / (np.abs(diff).max() + 1e-8)

        cur_t  = torch.FloatTensor(cur.transpose(2,0,1)).unsqueeze(0)   # (1,3,H,W)
        diff_t = torch.FloatTensor(dnorm.transpose(2,0,1)).unsqueeze(0)

        with torch.no_grad():
            bvp_val = self.model(cur_t, diff_t).item()
        self.bvp_buffer.append(bvp_val)

    def ready(self):
        return len(self.bvp_buffer) >= self.T

    def predict(self):
        """Convert accumulated BVP waveform → BPM.  Returns float or None."""
        if not self.ready():
            return None
        try:
            sig = np.array(list(self.bvp_buffer), dtype=np.float32)
            return self._to_bpm(sig)
        except Exception as e:
            print(f"DeepPhys inference error: {e}")
            return None

    def _to_bpm(self, sig):
        sig = sp_signal.detrend(sig)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        nyq = self.fps / 2.0
        b, a = sp_signal.butter(3, [0.7/nyq, 3.5/nyq], btype='band')
        try:
            sig = sp_signal.filtfilt(b, a, sig)
        except Exception:
            pass
        freqs = fftfreq(len(sig), d=1.0/self.fps)
        mag   = np.abs(fft(sig))
        valid = (freqs >= 0.7) & (freqs <= 3.5)
        if not valid.any():
            return None
        peak = freqs[valid][np.argmax(mag[valid])]
        bpm  = float(peak * 60.0)
        return bpm if 40 <= bpm <= 180 else None
