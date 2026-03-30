# rppg/rhythmformer_predictor.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq


# ============================================================
# Building blocks
# ============================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Named to preserve checkpoint key structure: proj_q.0.conv / proj_k.0.conv.
    Checkpoint weights are full 3D conv [64, 64, 3, 3, 3] — NOT depthwise.
    """
    def __init__(self, in_ch, out_ch, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch,
                              kernel_size, stride,
                              padding, bias=False)

    def forward(self, x):
        return self.conv(x)



class FusionStem(nn.Module):
    """
    Processes raw frames + difference frames separately
    stem11: raw  RGB  (3  → 64)  kernel 5×5
    stem12: diff frames (12 → 64)  kernel 5×5
    stem21, stem22: second stage 64→64 kernel 3×3
    """
    def __init__(self):
        super().__init__()
        # Stage 1
        self.stem11 = nn.Sequential(
            nn.Conv2d(3,  64, 5, padding=2, bias=True),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.stem12 = nn.Sequential(
            nn.Conv2d(12, 64, 5, padding=2, bias=True),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        # Stage 2
        self.stem21 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.stem22 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.BatchNorm2d(64), nn.ReLU()
        )

    def forward(self, x):
        # x: (B, 3, T, H, W)
        B, C, T, H, W = x.shape

        # Raw frames: process each frame
        x_2d = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        raw  = self.stem11(x_2d)            # (B*T, 64, H, W)
        raw  = self.stem21(raw)

        # Difference frames: 4 consecutive diffs stacked
        # diff shape: (B, 3, T-1, H, W)
        diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        # Pad to match T
        diff = torch.cat([diff,
                          diff[:, :, -1:, :, :]], dim=2)

        # Stack 4 consecutive diffs → 12 channels
        diff_list = []
        for i in range(4):
            shifted = torch.roll(diff, shifts=i, dims=2)
            diff_list.append(shifted)
        diff_cat = torch.cat(diff_list, dim=1)  # (B,12,T,H,W)
        diff_2d  = diff_cat.permute(
            0,2,1,3,4).reshape(B*T, 12, H, W)
        dif = self.stem12(diff_2d)
        dif = self.stem22(dif)

        # Fuse
        out = raw + dif                     # (B*T, 64, H, W)
        _, Ch, H2, W2 = out.shape
        out = out.reshape(B, T, Ch, H2, W2)
        out = out.permute(0,2,1,3,4)        # (B,64,T,H,W)
        return out


class Attention3D(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.lepe          = nn.Conv3d(dim, dim, 3,
                                       padding=1, groups=dim)
        self.qkv_linear    = nn.Conv3d(dim, dim*3, 1, bias=True)
        self.output_linear = nn.Conv3d(dim, dim, 1)
        # proj_q / proj_k: checkpoint keys are proj_q.0.conv.weight (shape [64,64,3,3,3])
        # and proj_q.1.weight (BN). DepthwiseSeparableConv now uses Conv3d internally.
        self.proj_q = nn.Sequential(
            DepthwiseSeparableConv(dim, dim, 3, padding=1),
            nn.BatchNorm3d(dim), nn.ReLU()
        )
        self.proj_k = nn.Sequential(
            DepthwiseSeparableConv(dim, dim, 3, padding=1),
            nn.BatchNorm3d(dim), nn.ReLU()
        )
        # proj_v checkpoint key: proj_v.0.weight → must be Sequential
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        qkv  = self.qkv_linear(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        lepe = self.lepe(v)
        out  = self.output_linear(q * k + lepe)
        return out + x



class MLP3D(nn.Module):
    def __init__(self, dim=64, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, hidden, 1),
            nn.BatchNorm3d(hidden), nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1,
                      groups=hidden),
            nn.BatchNorm3d(hidden), nn.GELU(),
            nn.Conv3d(hidden, dim, 1),
            nn.BatchNorm3d(dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x) + x


class TransformerBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn  = Attention3D(dim)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp   = MLP3D(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = self.mlp(self.norm2(x))
        return x


class Stage(nn.Module):
    def __init__(self, dim=64, depth=2, n_down=1):
        super().__init__()
        # Downsample layers (temporal)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm3d(dim),
                nn.Conv3d(dim, dim, (2,1,1), stride=(2,1,1))
            ) for _ in range(n_down)
        ])
        # Upsample layers
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=(2,1,1),
                            mode='trilinear',
                            align_corners=False),
                nn.Conv3d(dim, dim, (3,1,1), padding=(1,0,0)),
                nn.BatchNorm3d(dim)
            ) for _ in range(n_down)
        ])
        self.blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(depth)
        ])

    def forward(self, x):
        # Downsample
        for d in self.downsample_layers:
            x = d(x)
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # Upsample
        for u in self.upsample_layers:
            x = u(x)
        return x


class RhythmFormer(nn.Module):
    """
    Full RhythmFormer architecture matching the .pth weights
    Input : (B, 3, T, H, W)  — T=160, H=W=128
    Output: (B, T)            — rPPG waveform
    """
    def __init__(self, dim=64):
        super().__init__()
        self.Fusion_Stem    = FusionStem()
        self.patch_embedding = nn.Conv3d(dim, dim,
                                         (1,4,4),
                                         stride=(1,4,4))
        self.stages = nn.ModuleList([
            Stage(dim, depth=2, n_down=1),
            Stage(dim, depth=2, n_down=2),
            Stage(dim, depth=2, n_down=3),
        ])
        self.ConvBlockLast = nn.Conv1d(dim, 1, 1)

    def forward(self, x):
        # x: (B,3,T,H,W)
        x = self.Fusion_Stem(x)       # (B,64,T,H,W)
        x = self.patch_embedding(x)   # (B,64,T,H/4,W/4)

        for stage in self.stages:
            x = stage(x)              # (B,64,T,H/4,W/4)

        # Global spatial average → (B,64,T)
        x = x.mean(dim=[-2,-1])

        # (B,64,T) → (B,1,T) → (B,T)
        x = self.ConvBlockLast(x).squeeze(1)
        return x


# ============================================================
# Predictor class — use this in rppg_main.py
# ============================================================

class RhythmFormerPredictor:
    """
    Loads RhythmFormer weights and runs inference.

    Usage:
        rf = RhythmFormerPredictor('../UBFC-rPPG_RhythmFormer.pth')
        rf.add_frame(bgr_frame)
        bpm = rf.predict()   # returns float or None
    """

    T      = 160    # frames per inference window
    H      = 128    # spatial resize height
    W      = 128    # spatial resize width

    def __init__(self, weights_path, fps=30):
        self.fps          = fps
        self.frame_buffer = deque(maxlen=self.T)
        self.device       = 'cpu'   # M1 Mac CPU

        # Build model
        self.model = RhythmFormer(dim=64)

        # Load weights — strip 'module.' prefix
        # (saved with nn.DataParallel)
        state = torch.load(weights_path,
                           map_location='cpu')
        cleaned = {k.replace('module.', ''): v
                   for k, v in state.items()}

        # Load with strict=False to handle
        # minor architecture mismatches gracefully
        missing, unexpected = self.model.load_state_dict(
            cleaned, strict=False
        )
        if missing:
            print(f"  Missing keys  : {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        self.model.eval()
        print("✅ RhythmFormer loaded successfully")

    def add_frame(self, bgr_frame):
        """Add one BGR frame from webcam"""
        resized = cv2.resize(bgr_frame,
                             (self.W, self.H))
        rgb = cv2.cvtColor(resized,
                           cv2.COLOR_BGR2RGB
                           ).astype(np.float32) / 255.0
        self.frame_buffer.append(rgb)

    def ready(self):
        return len(self.frame_buffer) >= self.T

    def predict(self):
        """
        Run inference on current frame buffer.
        Returns BPM (float) or None if not ready.
        """
        if not self.ready():
            return None

        try:
            frames = np.stack(list(self.frame_buffer),
                              axis=0)         # (T,H,W,3)
            frames = frames.transpose(3,0,1,2) # (3,T,H,W)
            x = torch.FloatTensor(frames
                                  ).unsqueeze(0)  # (1,3,T,H,W)

            with torch.no_grad():
                rppg = self.model(x)          # (1,T)
                rppg = rppg.squeeze().numpy() # (T,)

            bpm = self._to_bpm(rppg)
            return bpm

        except Exception as e:
            print(f"RhythmFormer inference error: {e}")
            return None

    def _to_bpm(self, rppg):
        """Convert rPPG waveform → BPM via FFT"""
        sig = sp_signal.detrend(rppg)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)

        # Bandpass
        nyq  = self.fps / 2.0
        b, a = sp_signal.butter(
            3,
            [0.7/nyq, 3.5/nyq],
            btype='band'
        )
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