# rPPG Heart Rate Estimation from Webcam

This project implements a classical **remote photoplethysmography (rPPG)** pipeline to estimate heart rate (BPM) from live webcam video using facial skin color variations.

The system relies on MediaPipe facial landmarks and signal processing techniques instead of deep learning, making the pipeline interpretable, lightweight, and suitable for academic experimentation.

---

## Overview

Remote photoplethysmography (rPPG) extracts physiological signals by analyzing subtle skin color changes caused by blood flow. This implementation tracks stable facial regions and converts them into a cardiac signal using classical filtering and frequency analysis.

**Pipeline**

```
Webcam
→ Face landmarks
→ ROI extraction
→ RGB signal extraction
→ rPPG signal construction
→ Bandpass filtering
→ FFT / PSD
→ Heart rate estimation (BPM)
```

---

## Features

- Real-time webcam heart rate estimation
- Landmark-based anatomical ROI tracking
- Green-channel rPPG extraction
- Cardiac band filtering (0.7–4.0 Hz)
- FFT-based BPM estimation
- Temporal smoothing for stability
- No deep learning required (baseline pipeline)

---

## Model File (Required)

This project uses the MediaPipe Face Landmarker model:

```
face_landmarker.task
```

If the model file is not present, download it from:

👉 https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Place the file at:

```
rppg/face_landmarker.task
```

The pipeline expects the model in this location.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ShubhamY90/rppg_heart_rate_estimation_using_webcam.git
cd rppg_heart_rate_estimation_using_webcam
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script:

```bash
python webcam_test.py
```

For best results:

- Use good lighting
- Keep head movement minimal
- Ensure face is fully visible

---

## Project Structure

```
rppg/
  buffer.py
  chrom.py
  signal_processing.py
  face_landmarks.py
  forehead_roi.py
  forehead_and_cheeks.py
  face_landmarker.task

webcam_test.py
roi_demo.py
```

---

## Limitations

- Sensitive to lighting variations
- Motion artifacts affect signal quality
- Consumer webcams limit precision
- Not a medical device

---

## Future Work

- Deep learning temporal modeling (LSTM / Transformer)
- Motion-robust rPPG extraction
- Multi-ROI signal fusion
- Cross-dataset evaluation
- Real-time mobile deployment

---

## License

MIT License

---

## Author

Shubham Yadav  
Academic rPPG Research Project
