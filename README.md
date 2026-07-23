# rPPG Heart Rate Estimation via Webcam

> Real-time, contact-free heart rate measurement using Remote Photoplethysmography (rPPG) — directly from a standard webcam, powered by a 3-branch adaptive fusion pipeline combining classical signal processing, a custom-trained LSTM, and the RhythmFormer transformer model.

📄 **Technical Report:**  
**[Read the complete project report (PDF)](https://github.com/ShubhamY90/rppg_heart_rate_estimation_using_webcam/blob/main/rppg_heartrate_estimation.pdf)**


---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture — Fusion (`rppg_main_f.py`)](#pipeline-architecture--fusion-rppg_main_fpy)
3. [Signal-Processing-Only Pipeline (`rppg_main.py`)](#signal-processing-only-pipeline-rppg_mainpy)
4. [Models Used](#models-used)
   - [LSTM (custom-trained on UBFC)](#1-lstm-model--rppg_lstm_modelkeras)
   - [RhythmFormer Transformer](#2-rhythmformer-transformer--ubfc-rppg_rhythmformerpth)
5. [Face Detection — MediaPipe FaceMesh](#face-detection--mediapipe-facemesh)
6. [Signal Processing Pipeline](#signal-processing-pipeline)
7. [File-by-File Role Reference](#file-by-file-role-reference)
8. [Test Utilities](#test-utilities)
9. [Installation & Running](#installation--running)
   - [macOS](#macos)
   - [Windows](#windows)
10. [Known Notes & Tips](#known-notes--tips)

---


## Overview

This project estimates **heart rate (BPM) in real time** from webcam video without any contact sensor. It is based on the physiological principle that the subtle colour changes in skin caused by pulsating blood flow can be captured by a camera — this is called **remote photoplethysmography (rPPG)**.

The system runs **three independent branches** in parallel and dynamically fuses their outputs every second:

| Branch | Method | Model File |
|--------|--------|-----------|
| 1 | POS + CHROM signal processing + FFT | *(algorithmic — no weights)* |
| 2 | LSTM neural network | `rppg_lstm_model.keras` |
| 3 | RhythmFormer Vision Transformer | `UBFC-rPPG_RhythmFormer.pth` |

Dynamic weights are computed per-frame based on **signal-to-noise ratio (SNR)** and **branch agreement**, so the fusion adapts automatically to signal quality.

---

## Pipeline Architecture — Fusion (`rppg/rppg_main_f.py`)

This is the **primary, recommended entry point** and runs the full 3-branch fusion system.

```
Webcam Frame
     │
     ▼
[forehead_and_cheeks.py] ── MediaPipe FaceMesh (468 landmarks)
     │  Detects: forehead ROI, left cheek ROI, right cheek ROI
     │  Also: glasses detection, skin quality check, tilt correction
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
Forehead Buffer    Left Cheek Buffer  Right Cheek Buffer
(weight 0.60)      (weight 0.20)      (weight 0.20)
     │
     └──────────── combine_roi_signals() ──────────────────┐
                   Weighted POS signals from all 3 ROIs    │
                                                           ▼
                                              ┌────────────────────┐
                                              │  Combined rPPG sig │
                                              └────────────────────┘
                                                     │
                   ┌─────────────────────────────────┼──────────────────────────┐
                   ▼                                 ▼                          ▼
          [BRANCH 1]                         [BRANCH 2]                 [BRANCH 3]
     POS + FFT  &  CHROM + FFT           LSTM Predictor             RhythmFormer
     signal_processing.py               model_predictor.py         rhythmformer_predictor.py
     → bpm_signal, bpm_chrom            → bpm_lstm                 → bpm_rf
                   │                                 │                          │
                   └─────────────────────────────────┼──────────────────────────┘
                                                     ▼
                                        [fusion.py] dynamic_fusion_3branch()
                                        SNR-based adaptive weights → bpm_fused
                                                     │
                                                     ▼
                                         Final BPM + Confidence + Stress
                                         Displayed on live webcam feed
```

### Key novelties implemented

| Feature | Description |
|---------|-------------|
| **Multi-ROI weighted combination** | Forehead (0.60), left cheek (0.20), right cheek (0.20) — independently buffered then combined with POS |
| **Dual signal algorithms** | POS (Wang et al., 2017) *and* CHROM (De Haan & Jeanne, 2013) run in parallel in Branch 1 |
| **Harmonic correction** | If FFT peak ≈ 2× or 0.5× the previous BPM, it is halved/doubled to avoid octave errors |
| **Per-branch smoothing** | Each branch has its own temporal smoothing window + outlier rejection before fusion |
| **3-branch SNR-adaptive fusion** | Weights re-calculated every second from live SNR; branches that agree get bonus weight |
| **Confidence score** | SNR quality + inter-branch spread → `0–1` confidence with HIGH / MEDIUM / LOW label |
| **Stress indicator** | 30 s baseline HR acquired silently; current elevation mapped to `0–100` stress score |
| **Glasses detection** | Eye-region non-skin ratio flags glasses presence (reduces accuracy warning) |
| **Live rPPG waveform** | Mini oscilloscope panel drawn in bottom-right of the video feed |

### On-screen HUD

```
BPM: 72
--- Branches ---
POS+FFT  : 71 BPM  w=0.42
CHROM+FFT: 70 BPM
LSTM     : 73 BPM  w=0.31
RhythmF  : 72 BPM  w=0.27
─────────────────────────
SNR      : 0.312
Conf     : 0.78 [HIGH]
Stress   : 12/100 [LOW]
```

---

## Signal-Processing-Only Pipeline (`rppg/rppg_main.py`)

This is a **lightweight alternative** for anyone who wants to test rPPG without loading neural network models. It uses **only the forehead ROI** and runs the POS algorithm + bandpass filter + FFT.

Use this if:
- You don't have PyTorch / TensorFlow installed
- You want to validate that the webcam and face detection are working before running the full pipeline
- You are debugging signal quality

```
Webcam → MediaPipe FaceMesh → Forehead mask → RG B mean
      → POS algorithm → normalize → bandpass → FFT → BPM
```

Run it with:
```bash
python rppg/rppg_main.py
```

---

## Models Used

### 1. LSTM Model — `rppg_lstm_model.keras`

| Property | Detail |
|----------|--------|
| **Framework** | TensorFlow / Keras |
| **Architecture** | LSTM (Long Short-Term Memory) recurrent neural network |
| **Trained on** | UBFC-rPPG dataset (University of Burgundy Franche-Comté) |
| **Input** | 150-sample normalized rPPG signal window (shape `[1, 150, 1]`) |
| **Output** | Single BPM value (regression) |
| **File** | `rppg_lstm_model.keras` (root of project) |

The LSTM is trained to learn **temporal patterns in the rPPG waveform** that correspond to a given heart rate. It receives the last 150 samples of the combined, filtered rPPG signal (normalized to zero-mean, unit-variance), and directly regresses the BPM.  
This model was trained by us on the UBFC-rPPG benchmark dataset, which provides GT heart rates synchronized with RGB video from a webcam.

**Wrapper:** `rppg/model_predictor.py` → `LSTMPredictor` class

---

### 2. RhythmFormer Transformer — `UBFC-rPPG_RhythmFormer.pth`

| Property | Detail |
|----------|--------|
| **Framework** | PyTorch |
| **Architecture** | RhythmFormer — a spatiotemporal Vision Transformer for rPPG |
| **Trained on** | UBFC-rPPG dataset |
| **Input** | 160 consecutive webcam frames, each resized to 128×128 RGB (shape `[1, 3, 160, 128, 128]`) |
| **Output** | Waveform of length 160; converted to BPM via FFT |
| **File** | `UBFC-rPPG_RhythmFormer.pth` (root of project) |
| **Warmup** | Requires 160 frames (≈5 s) before the first prediction |

#### How RhythmFormer works

RhythmFormer is a hierarchical spatiotemporal transformer architecture:

1. **FusionStem** — processes raw RGB frames (3 → 64 ch) and temporal *difference* frames (4 consecutive diffs stacked → 12 → 64 ch) in parallel, then fuses them by addition. This gives the model sensitivity to both absolute colour and subtle frame-to-frame changes caused by blood flow.

2. **Patch embedding** — a `1×4×4` 3D convolution reduces spatial resolution (`H×W → H/4×W/4`), producing a compact spatiotemporal token grid.

3. **Three Transformer Stages** — each stage contains:
   - *Downsample* layers (temporal stride-2 convolution)
   - *Attention3D* blocks (local 3D attention with LEPE positional encoding)
   - *MLP3D* blocks (channel-mixing FFN with GELU activations)
   - *Upsample* layers (trilinear upsampling back to original temporal length)

4. **Global spatial average pooling** → `(B, 64, T)` → `Conv1d(64→1)` → rPPG waveform of length T.

5. **BPM estimation** — the output waveform is detrended, normalized, bandpass-filtered (0.7–3.5 Hz), and the dominant FFT peak in `[0.7, 3.5]` Hz is converted to BPM.

**Wrapper:** `rppg/rhythmformer_predictor.py` → `RhythmFormerPredictor` class

---

## Face Detection — MediaPipe FaceMesh

The project uses **MediaPipe FaceLandmarker** (Tasks API), which runs in two phases:

1. **BlazeFace** — a lightweight single-shot face detector that finds bounding boxes with sub-millisecond latency. This is the *detection* phase.

2. **Deep Landmark Network** — once a face is found, a second model runs a dense mesh prediction producing **468 3D facial landmarks** that accurately segment key facial geometry.

### What we use from the 468 landmarks

| Usage | Landmarks |
|-------|-----------|
| Eye line / tilt angle | `#33` (left eye corner), `#263` (right eye corner) |
| Face width reference | `#127` (left jaw), `#356` (right jaw) |
| Face height reference | `#152` (chin), eye midpoint |
| Forehead top | `#10` |
| Cheek positions | `#266`, `#323` (left), `#36`, `#93` (right) |
| Glasses detection | Eye-outline landmarks `#33, 160, 159…` (left) + `#263, 387, 386…` (right) |

### ROI construction

- Landmarks are used to compute **face width, face height, tilt angle, and anchor positions**.
- Each ROI (forehead, left cheek, right cheek) is an **oriented (rotated) bounding box** aligned to the detected face angle, so tilted faces are handled correctly.
- ROI sizes are relative to face scale (e.g. forehead = 45% face width × 25% face height).
- A **hybrid HSV + YCrCb skin detection** is applied inside each ROI to filter out hair, glasses, and background pixels.
- If the skin coverage falls below a threshold, alerts are shown and measurement is paused.

**Model file:** `rppg/face_landmarker.task` (auto-downloaded if missing)

---

## Signal Processing Pipeline

The signal processing used within Branch 1 follows these steps in order:

```
Raw ROI pixels
    │
    ▼  1. Mean RGB per frame over mask pixels
    ▼  2. POS algorithm (Wang et al. 2017)
       X  = Xn[G] - Xn[B]
       Y  = Xn[R] + Xn[G] - 2·Xn[B]
       α  = std(X) / std(Y)
       sig = X - α·Y
    │
    ▼  3. Detrending (scipy.signal.detrend — removes slow drift)
    ▼  4. Z-score normalization (zero mean, unit variance)
    ▼  5. Butterworth Bandpass Filter (0.75–3.0 Hz = 45–180 BPM)
       Order 3, zero-phase via filtfilt
    │
    ▼  6. FFT → find dominant frequency in [0.75, 3.0] Hz range
    ▼  7. Convert Hz → BPM  (BPM = peak_freq × 60)
    ▼  8. SNR = peak_power² / total_band_power²
    │
    ▼  9. Harmonic correction (halve if ≈2× prev, double if ≈0.5× prev)
    ▼ 10. Per-branch temporal smoothing (6-frame rolling mean, outlier rejection)
    │
    ▼ → bpm_signal
```

**CHROM** (De Haan & Jeanne, 2013) runs in parallel on the forehead-only buffer:
```
Xs = 3·Rn - 2·Gn
Ys = 1.5·Rn + Gn - 1.5·Bn
α  = std(Xs) / std(Ys)
sig = Xs - α·Ys
```
CHROM is more robust to motion artifacts and provides a secondary estimate (`bpm_chrom`) shown on the HUD but not fused.

### Dynamic Fusion Weights (`rppg/fusion.py`)

```
SNR  (computed from combined_sig FFT)
      │
      ▼
snr_norm = clip(SNR / 0.3, 0, 1)

w_SP   = 0.25 + 0.25 × snr_norm      # ↑ when signal is clean
w_LSTM = 0.25 + 0.10 × (1−snr_norm)  # ↑ when signal is noisy
w_RF   = 0.50 − 0.35 × snr_norm      # dominant when signal is weak

If RF and SP agree (|bpm_rf − bpm_sp| < 5):
    w_SP  += 0.10
    w_RF  += 0.10
    w_LSTM = max(0.10, w_LSTM − 0.20)

If RF not yet warmed up:
    extra weight redistributed: 60% → SP, 40% → LSTM

Normalize so w_SP + w_LSTM + w_RF = 1
BPM_final = w_SP·SP + w_LSTM·LSTM + w_RF·RF
```

---

## File-by-File Role Reference

```
ROI/
├── rppg/
│   ├── rppg_main_f.py          ← MAIN: Full 3-branch fusion pipeline (run this)
│   ├── rppg_main.py            ← ALT:  Signal-processing-only (no neural nets)
│   ├── forehead_and_cheeks.py  ← Face detection + ROI extraction via MediaPipe
│   ├── buffer.py               ← SignalBuffer: rolling deque for R, G, B channels
│   ├── signal_processing.py    ← bandpass_filter() + estimate_bpm() utilities
│   ├── chrom.py                ← chrom_signal() — standalone CHROM algorithm
│   ├── model_predictor.py      ← LSTMPredictor: loads .keras model, runs inference
│   ├── rhythmformer_predictor.py ← RhythmFormerPredictor + full model definition
│   ├── fusion.py               ← dynamic_fusion_3branch(): SNR-adaptive weighting
│   └── face_landmarker.task    ← MediaPipe binary model (auto-downloaded)
│
├── rppg_lstm_model.keras       ← Trained LSTM weights (TensorFlow/Keras)
├── UBFC-rPPG_RhythmFormer.pth  ← Trained RhythmFormer weights (PyTorch)
│
├── inspect_model.py            ← TEST: print RhythmFormer checkpoint keys/shapes
├── webcam_test.py              ← TEST: verify webcam opens and streams
├── roi_demo.py                 ← TEST: live forehead ROI box using MediaPipe
│
└── requirements.txt            ← Full dependency list
```

### Module details

| File | Role |
|------|------|
| `rppg_main_f.py` | Entry point for the full system. Opens webcam, orchestrates ROI extraction, fills buffers, runs all three branches, calls fusion, computes confidence + stress, draws HUD and waveform. |
| `rppg_main.py` | Simplified entry point. Only forehead ROI, only POS algorithm, no neural network models. Good for debugging. |
| `forehead_and_cheeks.py` | Loads MediaPipe FaceLandmarker. Provides `get_rois(frame)` which returns dict of binary masks for forehead, left cheek, right cheek; skin quality ratios; glasses flag; alert messages. |
| `buffer.py` | `SignalBuffer` stores rolling windows of mean R, G, B values per ROI. `ready()` returns True once filled to `window_seconds × fps` samples. |
| `signal_processing.py` | `bandpass_filter()` — 3rd-order Butterworth zero-phase filter. `estimate_bpm()` — FFT-based peak frequency → BPM. |
| `chrom.py` | `chrom_signal()` — standalone CHROM rPPG extraction. Also used inline inside `rppg_main_f.py`. |
| `model_predictor.py` | `LSTMPredictor` loads `rppg_lstm_model.keras` and exposes `predict(signal)` → BPM float. |
| `rhythmformer_predictor.py` | Defines the full RhythmFormer architecture (FusionStem, Attention3D, MLP3D, TransformerBlock, Stage, RhythmFormer). `RhythmFormerPredictor` manages the frame buffer, loads weights, and exposes `add_frame()` / `predict()`. |
| `fusion.py` | `dynamic_fusion_3branch()` — computes live SNR from the filtered signal, sets adaptive weights for all three branches, handles missing RhythmFormer, normalizes weights, returns fused BPM + per-branch weights + SNR. |

---

## Test Utilities

Before running the full pipeline, use these three scripts to verify your environment step-by-step.

### 1. `inspect_model.py` — Verify RhythmFormer Checkpoint

```bash
python inspect_model.py
```

Loads `UBFC-rPPG_RhythmFormer.pth` with PyTorch and prints the top-level keys, tensor shapes, and any nested dict structure. Use this to confirm:
- The `.pth` file downloaded correctly and is not corrupted
- The checkpoint structure matches what `rhythmformer_predictor.py` expects (keys prefixed with `module.` get stripped automatically)

---

### 2. `webcam_test.py` — Verify Webcam Access

```bash
python webcam_test.py
```

Opens webcam index `0` (default), displays a live feed in a window. If the window doesn't open or you see `ERROR: Webcam cannot be opened`, fix your camera driver / permissions before running the main pipeline. Press **Q** to quit.

---

### 3. `roi_demo.py` — Verify MediaPipe Face Detection & ROI

```bash
python roi_demo.py
```

Loads the MediaPipe FaceLandmarker and draws a **forehead bounding rectangle** in real time. Use this to confirm:
- MediaPipe is installed and the model file is accessible
- Your face is being detected and landmarks are stable
- The ROI is roughly in the right position on your forehead

Press **Q** to quit.

---

## Installation & Running

### macOS

**Recommended Python: 3.10 or 3.11** (TensorFlow and MediaPipe are best tested here).

#### 1. Clone the repository

```bash
git clone https://github.com/ShubhamY90/rppg_heart_rate_estimation_using_webcam

```

#### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install core dependencies

```bash
pip install --upgrade pip
pip install opencv-python mediapipe tensorflow torch torchvision scipy numpy
```

> **Protobuf note (macOS):** If you see an `ImportError` about protobuf, run:
> ```bash
> pip install "protobuf>=3.20,<4" --upgrade
> ```

#### 4. Run the test utilities (recommended first-time check)

```bash
python webcam_test.py          # confirm webcam works
python inspect_model.py        # confirm PyTorch checkpoint loads
python roi_demo.py             # confirm MediaPipe detects your face
```

#### 5. Run the full fusion pipeline

```bash
python rppg/rppg_main_f.py
```

#### 5b. (Optional) Run the signal-processing-only pipeline

```bash
python rppg/rppg_main.py
```

#### Camera permissions on macOS

Go to **System Settings → Privacy & Security → Camera** and make sure your terminal app (Terminal / iTerm2 / VS Code) is allowed.

---

### Windows

**Recommended Python: 3.10** (install from [python.org](https://python.org) — check *Add to PATH*).

#### 1. Clone the repository

```cmd
git clone https://github.com/ShubhamY90/rppg_heart_rate_estimation_using_webcam

```

#### 2. Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Install core dependencies

```cmd
pip install --upgrade pip
pip install opencv-python mediapipe tensorflow torch torchvision scipy numpy
```

> **Windows-specific notes:**
> - `tensorflow` on Windows requires **Microsoft Visual C++ Redistributable** — download from Microsoft if not already installed.
> - If you have an NVIDIA GPU and want GPU acceleration for PyTorch, install the CUDA-enabled build:
>   ```cmd
>   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
>   ```
> - If you see a protobuf error, run:
>   ```cmd
>   pip install "protobuf>=3.20,<4" --upgrade
>   ```

#### 4. Run the test utilities

```cmd
python webcam_test.py
python inspect_model.py
python roi_demo.py
```

#### 5. Run the full fusion pipeline

```cmd
python rppg/rppg_main_f.py
```

#### 5b. (Optional) Run the signal-processing-only pipeline

```cmd
python rppg/rppg_main.py
```

#### Camera permissions on Windows

Windows may show a camera permission dialog the first time. If the camera access is blocked, go to **Settings → Privacy & Security → Camera** and enable access for your terminal / IDE.

---

## Known Notes & Tips

| Situation | Tip |
|-----------|-----|
| **RhythmFormer shows "warming (X/160)"** | Normal — it needs 160 frames (~5 s) before its first prediction. |
| **BPM jumps wildly at start** | The 15 s signal buffer must fill first. The progress bar at the bottom shows warmup status. |
| **Glasses detected warning** | Glasses absorb or reflect light and reduce accuracy. Remove them or accept reduced precision. |
| **"Move hair from forehead"** | Hair inside the forehead ROI introduces noise. Pull hair back or adjust your position. |
| **Low SNR (red indicator)** | Improve lighting — a bright, even, front-facing light source dramatically improves signal quality. |
| **Stress indicator locked at "calibrating"** | The stress baseline takes 30 s of readings to establish. Sit still for the first 30 s. |
| **Face not detected on tilt** | FaceMesh handles moderate tilt. Extreme angles (>45°) may cause detection loss. |
| **macOS M1/M2 chips** | RhythmFormer runs on CPU by default (`device='cpu'`). MPS (Metal) support can be enabled in `rhythmformer_predictor.py` if needed. |
| **Script path issues** | Run scripts from the project root (`ROI/`) — the path setup in each file uses relative imports anchored to `__file__`. |

---

## References

- Wang, W. et al. (2017). *Algorithmic Principles of Remote PPG*. IEEE TBME.
- De Haan, G. & Jeanne, V. (2013). *Robust Pulse Rate From Chrominance-Based rPPG*. IEEE TBME.
- RhythmFormer: *Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer*. (UBFC-rPPG weights used with permission / public release).
- X. Zhang, Z. Zhang, Y. Wang, M. Wang and B. W. -K. Ling, "An End-to-End Non-Contact Heart Rate Estimation Method Based on Facial        Videos via Continuous Tracking Model," in IEEE Transactions on Consumer Electronics, vol. 72, no. 1, pp. 1205-1207, Feb. 2026, doi:     10.1109/TCE.2025.3650004.
keywords: {Heart rate;Feature extraction;Estimation;Videos;Skin;Monitoring;Training;Lighting;Facial features;Accuracy;rPPG;heart rate estimation;ROI detection;continuous tracking},


- UBFC-rPPG Dataset: Bobbia et al., *Unsupervised skin tissue segmentation for remote photoplethysmography*, Pattern Recognition Letters, 2019.
- MediaPipe FaceLandmarker: Google LLC. https://developers.google.com/mediapipe
