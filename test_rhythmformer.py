"""
test_rhythmformer.py
====================
Evaluates the RhythmFormer model (arxiv: 2402.12788) on the UBFC-rPPG dataset.

Paper: "RhythmFormer: Extracting Patterned rPPG Signals based on
         Periodic Sparse Attention"  — Zou et al., Pattern Recognition 2025.

Evaluation protocol (as per the paper):
  ─ Input clips  : T=160 consecutive face frames, resized to 128×128 RGB
  ─ Stride       : non-overlapping windows (stride = T)
  ─ HR estimation: FFT peak in [0.7, 3.5] Hz → BPM = peak_Hz × 60
  ─ Ground truth : mean HR over each clip window (from UBFC gt file)
  ─ Metrics      : MAE (BPM) · RMSE (BPM) · Pearson r

UBFC-rPPG dataset folder structure expected:
  <dataset_root>/
      subject1/
          vid.avi
          ground_truth.txt      ← col-0: rPPG signal  col-1: timestamp(ms)
                                   col-2: HR(BPM)
      subject2/ ...

Usage:
  python test_rhythmformer.py \
      --dataset   /path/to/UBFC-rPPG \
      --weights   UBFC-rPPG_RhythmFormer.pth \
      [--subjects subject1 subject2 ...]   # optional filter
      [--stride   160]                     # clip stride (default = T = non-overlap)
      [--device   cpu]
"""

import os
import sys
import argparse
import glob
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr

# ──────────────────────────────────────────────────────────────────────────────
# Import RhythmFormer architecture from the existing predictor module
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rppg"))
from rhythmformer_predictor import RhythmFormer  # architecture only


# ══════════════════════════════════════════════════════════════════════════════
# Constants (must match the paper / pre-trained checkpoint)
# ══════════════════════════════════════════════════════════════════════════════
T_CLIP  = 160        # frames per inference window
H, W    = 128, 128   # spatial resolution
HR_MIN  = 40.0       # BPM lower bound
HR_MAX  = 180.0      # BPM upper bound
FREQ_LO = 0.7        # Hz  (40 BPM)
FREQ_HI = 3.5        # Hz  (210 BPM)


# ══════════════════════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════════════════════
def load_model(weights_path: str, device: str) -> torch.nn.Module:
    """Load pre-trained RhythmFormer weights into the model."""
    model = RhythmFormer(dim=64)
    state = torch.load(weights_path, map_location="cpu")

    # Strip DataParallel 'module.' prefix if present
    cleaned = {}
    for k, v in state.items():
        new_key = k.replace("module.", "")
        cleaned[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in checkpoint")

    model.eval()
    model.to(device)
    print(f"✅  RhythmFormer loaded  →  device={device}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    """
    frames : (T, H, W, 3)  uint8 RGB
    returns: (1, 3, T, H, W) float32 tensor in [0, 1]
    """
    x = frames.astype(np.float32) / 255.0     # (T,H,W,3)
    x = x.transpose(3, 0, 1, 2)              # (3,T,H,W)
    x = torch.from_numpy(x).unsqueeze(0)     # (1,3,T,H,W)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# HR estimation from rPPG waveform (identical to the paper's FFT approach)
# ══════════════════════════════════════════════════════════════════════════════
def rppg_to_hr(rppg: np.ndarray, fps: float) -> float:
    """
    Convert a raw rPPG waveform to a heart rate in BPM via FFT.

    Steps (from Section III-D of the paper):
      1. Detrend  (remove linear trend)
      2. Normalise (zero-mean, unit-std)
      3. Butterworth bandpass [FREQ_LO, FREQ_HI] Hz
      4. FFT peak frequency → BPM
    """
    sig = sp_signal.detrend(rppg.copy())
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    # Bandpass filter
    nyq  = fps / 2.0
    low  = FREQ_LO / nyq
    high = FREQ_HI / nyq
    # Clamp to valid range
    low  = max(low,  1e-4)
    high = min(high, 1.0 - 1e-4)
    b, a = sp_signal.butter(3, [low, high], btype="band")
    try:
        sig = sp_signal.filtfilt(b, a, sig)
    except Exception:
        pass  # skip filter if signal too short

    freqs = fftfreq(len(sig), d=1.0 / fps)
    mag   = np.abs(fft(sig))

    valid  = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    if not valid.any():
        return float("nan")

    peak_hz = freqs[valid][np.argmax(mag[valid])]
    bpm     = float(peak_hz * 60.0)

    if not (HR_MIN <= bpm <= HR_MAX):
        return float("nan")
    return bpm


# ══════════════════════════════════════════════════════════════════════════════
# UBFC-rPPG ground-truth reader
# ══════════════════════════════════════════════════════════════════════════════
def load_ubfc_gt(gt_path: str):
    """
    Parse UBFC-rPPG ground_truth.txt.
    Returns: gt_hr  — 1-D array of HR(BPM) per video frame.
    File columns: rPPG_signal  timestamp(ms)  HR_BPM
    """
    data = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append(float(parts[2]))   # HR column
                except ValueError:
                    pass
    if not data:
        raise RuntimeError(f"Could not parse HR from {gt_path}")
    return np.array(data, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Per-subject evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_subject(subj_dir: str, model: torch.nn.Module,
                     device: str, stride: int) -> list:
    """
    Returns a list of (pred_bpm, gt_bpm) tuples for all valid clips in one subject.
    """
    vid_path = os.path.join(subj_dir, "vid.avi")
    gt_path  = os.path.join(subj_dir, "ground_truth.txt")

    if not os.path.exists(vid_path):
        print(f"  [skip] no vid.avi in {subj_dir}")
        return []
    if not os.path.exists(gt_path):
        print(f"  [skip] no ground_truth.txt in {subj_dir}")
        return []

    # ── Read video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (W, H))
        all_frames.append(frame_rgb)
    cap.release()

    if len(all_frames) < T_CLIP:
        print(f"  [skip] {subj_dir}: only {len(all_frames)} frames (need {T_CLIP})")
        return []

    all_frames = np.stack(all_frames, axis=0)   # (N, H, W, 3)
    N = len(all_frames)

    # ── Read GT ───────────────────────────────────────────────────────────────
    gt_hr = load_ubfc_gt(gt_path)
    # Align length to video frames (GT may differ slightly)
    gt_hr = gt_hr[:N] if len(gt_hr) >= N else \
            np.pad(gt_hr, (0, N - len(gt_hr)), mode="edge")

    # ── Sliding-window inference ───────────────────────────────────────────────
    pairs = []
    start = 0
    while start + T_CLIP <= N:
        end    = start + T_CLIP
        clip   = all_frames[start:end]          # (T, H, W, 3)
        gt_clip = gt_hr[start:end]

        # Model inference
        x = preprocess_frames(clip).to(device)
        with torch.no_grad():
            rppg = model(x).squeeze().cpu().numpy()  # (T,)

        pred_bpm = rppg_to_hr(rppg, fps)
        gt_bpm   = float(np.nanmean(gt_clip))

        if not np.isnan(pred_bpm) and not np.isnan(gt_bpm):
            pairs.append((pred_bpm, gt_bpm))

        start += stride

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate metrics (as reported in paper Table I/II)
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(pairs: list) -> dict:
    """
    pairs: list of (pred_bpm, gt_bpm)
    Returns dict with MAE, RMSE, Pearson r.
    """
    if not pairs:
        return {"MAE": float("nan"), "RMSE": float("nan"), "Pearson_r": float("nan")}

    preds = np.array([p[0] for p in pairs])
    gts   = np.array([p[1] for p in pairs])

    errors = preds - gts
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))

    if len(preds) > 1:
        r, _ = pearsonr(preds, gts)
    else:
        r = float("nan")

    return {"MAE": mae, "RMSE": rmse, "Pearson_r": r}


# ══════════════════════════════════════════════════════════════════════════════
# Pretty print
# ══════════════════════════════════════════════════════════════════════════════
def print_metrics(metrics: dict, label: str = "Overall"):
    """Print a neatly formatted metrics table."""
    sep = "─" * 50
    print(f"\n{'═' * 50}")
    print(f"  RhythmFormer  ·  {label}")
    print(f"  (arxiv: 2402.12788 — Zou et al., 2025)")
    print(f"{'═' * 50}")
    print(f"  {'Metric':<20}{'Value':>15}")
    print(sep)
    print(f"  {'MAE  (BPM)':<20}{metrics['MAE']:>14.4f}")
    print(f"  {'RMSE (BPM)':<20}{metrics['RMSE']:>14.4f}")
    print(f"  {'Pearson r':<20}{metrics['Pearson_r']:>14.4f}")
    print(f"{'═' * 50}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Test RhythmFormer on UBFC-rPPG  (arxiv 2402.12788)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to the UBFC-rPPG root directory"
    )
    parser.add_argument(
        "--weights", type=str,
        default=os.path.join(os.path.dirname(__file__),
                             "UBFC-rPPG_RhythmFormer.pth"),
        help="Path to the pre-trained .pth checkpoint (default: repo root)"
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject folder names to evaluate (default: all)"
    )
    parser.add_argument(
        "--stride", type=int, default=T_CLIP,
        help=f"Clip stride in frames [default: {T_CLIP} = non-overlapping]"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device  [cpu | cuda | mps]"
    )
    args = parser.parse_args()

    # ── Discover subjects ─────────────────────────────────────────────────────
    dataset_root = os.path.abspath(args.dataset)
    if not os.path.isdir(dataset_root):
        print(f"[error] Dataset directory not found: {dataset_root}")
        sys.exit(1)

    if args.subjects:
        subj_dirs = [os.path.join(dataset_root, s) for s in args.subjects]
    else:
        subj_dirs = sorted([
            d for d in glob.glob(os.path.join(dataset_root, "subject*"))
            if os.path.isdir(d)
        ])
        if not subj_dirs:
            # Fall back: any immediate subdirectory
            subj_dirs = sorted([
                os.path.join(dataset_root, d)
                for d in os.listdir(dataset_root)
                if os.path.isdir(os.path.join(dataset_root, d))
            ])

    if not subj_dirs:
        print(f"[error] No subject folders found in {dataset_root}")
        sys.exit(1)

    print(f"\n📂  Dataset   : {dataset_root}")
    print(f"📄  Weights   : {args.weights}")
    print(f"🔢  Subjects  : {len(subj_dirs)}")
    print(f"🖼   Clip size : T={T_CLIP}, H={H}, W={W}")
    print(f"⏩  Stride    : {args.stride} frames")
    print(f"💻  Device    : {args.device}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.weights, args.device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_pairs    = []
    subj_results = {}

    for subj_dir in subj_dirs:
        name = os.path.basename(subj_dir)
        print(f"  Processing  {name} ...", end="  ", flush=True)
        pairs = evaluate_subject(subj_dir, model, args.device, args.stride)
        if pairs:
            m = compute_metrics(pairs)
            subj_results[name] = m
            all_pairs.extend(pairs)
            print(f"clips={len(pairs)}  "
                  f"MAE={m['MAE']:.2f}  "
                  f"RMSE={m['RMSE']:.2f}  "
                  f"r={m['Pearson_r']:.4f}")
        else:
            print("no valid clips — skipped")

    # ── Per-subject table ─────────────────────────────────────────────────────
    if subj_results:
        print("\n" + "─" * 70)
        print(f"  {'Subject':<14}{'MAE':>10}{'RMSE':>10}{'Pearson r':>12}")
        print("─" * 70)
        for name, m in subj_results.items():
            print(f"  {name:<14}{m['MAE']:>10.4f}{m['RMSE']:>10.4f}"
                  f"{m['Pearson_r']:>12.4f}")
        print("─" * 70)

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall = compute_metrics(all_pairs)
    print_metrics(overall,
                  label=f"UBFC-rPPG  ({len(all_pairs)} clips / "
                        f"{len(subj_results)} subjects)")

    # ── Return for programmatic use ───────────────────────────────────────────
    return overall


if __name__ == "__main__":
    main()
