# rppg/physnet_predictor.py
# PhysNet: Remote Physiological Measurement with 3D Attention Network
# (Yu et al., 2019, BMVC) — PyTorch implementation.
#
# Architecture: 3D CNN encoder–decoder
#   Input : (B, 3, T, H, W)  frames (T=32, H=W=32 by default)
#   Output: (B, T)            rPPG waveform, one scalar per frame
#
# The predictor accumulates incoming webcam frames in a sliding window,
# runs inference, and converts the output waveform to BPM via FFT.
#
# Weights: pass weights_path=None to run with random initialisation.

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(1,1,1), stride=(1,1,1), pad=(0,0,0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ELU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class PhysNet(nn.Module):
    """
    Lightweight PhysNet — encoder then 1D conv head.
    Reference architecture follows Yu et al. 2019 with
    channel counts scaled down for CPU inference.

    Input : (B, 3, T, H, W)   T=32, H=W=32
    Output: (B, T)
    """
    def __init__(self, T=32):
        super().__init__()
        self.encoder = nn.Sequential(
            # Stage 1  (3 → 16)
            ConvBlock3D(3,  16, (1,5,5), pad=(0,2,2)),
            ConvBlock3D(16, 16, (3,3,3), pad=(1,1,1)),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),   # H/2

            # Stage 2  (16 → 32)
            ConvBlock3D(16, 32, (3,3,3), pad=(1,1,1)),
            ConvBlock3D(32, 32, (3,3,3), pad=(1,1,1)),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),   # T/2, H/4

            # Stage 3  (32 → 64)
            ConvBlock3D(32, 64, (3,3,3), pad=(1,1,1)),
            ConvBlock3D(64, 64, (3,3,3), pad=(1,1,1)),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),   # T/4, H/8

            # Stage 4  (64 → 64)
            ConvBlock3D(64, 64, (3,3,3), pad=(1,1,1)),
            ConvBlock3D(64, 64, (3,3,3), pad=(1,1,1)),
        )
        # Collapse spatial → per-frame scalar
        self.pool_spatial = nn.AdaptiveAvgPool3d((None, 1, 1))   # (B,64,T',1,1)
        self.head         = nn.Conv1d(64, 1, 1)                  # (B,1,T')
        self.upsample     = nn.Upsample(size=T, mode='linear',
                                        align_corners=False)
        self.T = T

    def forward(self, x):
        # x: (B, 3, T, H, W)
        feat = self.encoder(x)                    # (B,64,T',H',W')
        feat = self.pool_spatial(feat)            # (B,64,T',1,1)
        feat = feat.squeeze(-1).squeeze(-1)       # (B,64,T')
        out  = self.head(feat)                    # (B,1,T')
        out  = self.upsample(out)                 # (B,1,T)
        return out.squeeze(1)                     # (B,T)


# ──────────────────────────────────────────────────────────────────────────────
# Predictor wrapper
# ──────────────────────────────────────────────────────────────────────────────

class PhysNetPredictor:
    """
    Sliding-window PhysNet predictor.

    Usage:
        pn = PhysNetPredictor(weights_path='physnet.pth')  # or None
        pn.add_frame(bgr_frame)
        bpm = pn.predict()   # float or None
    """

    T = 32     # frames per inference window
    H = 32     # spatial resize
    W = 32

    def __init__(self, weights_path=None, fps=30):
        self.fps          = fps
        self.frame_buffer = deque(maxlen=self.T)

        self.model = PhysNet(T=self.T)
        self.model.eval()

        if weights_path and os.path.isfile(weights_path):
            try:
                state = torch.load(weights_path, map_location='cpu')
                state = {k.replace('module.', ''): v for k, v in state.items()}
                self.model.load_state_dict(state, strict=False)
                print(f"✅ PhysNet weights loaded from {weights_path}")
            except Exception as e:
                print(f"⚠️  PhysNet weight loading failed: {e}  (using random init)")
        else:
            print("⚠️  PhysNet: no weights found — running with random init")

    def add_frame(self, bgr_frame):
        resized = cv2.resize(bgr_frame, (self.W, self.H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.frame_buffer.append(rgb)

    def ready(self):
        return len(self.frame_buffer) >= self.T

    def predict(self):
        """Returns BPM (float) or None."""
        if not self.ready():
            return None
        try:
            frames = np.stack(list(self.frame_buffer), axis=0)   # (T,H,W,3)
            frames = frames.transpose(3, 0, 1, 2)                # (3,T,H,W)
            x = torch.FloatTensor(frames).unsqueeze(0)           # (1,3,T,H,W)
            with torch.no_grad():
                rppg = self.model(x).squeeze().numpy()           # (T,)
            return self._to_bpm(rppg)
        except Exception as e:
            print(f"PhysNet inference error: {e}")
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
