import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================
# PARAMETERS (PHASE-1, DO NOT OVERTHINK)
# ==============================
SKIN_ALERT_THRESHOLD = 0.60   # below this → insufficient exposed skin

# ==============================
# Load Face Landmarker
# ==============================
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ==============================
# Webcam
# ==============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        cv2.putText(
            frame,
            "Face not detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        cv2.imshow("Forehead ROI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    lm = result.face_landmarks[0]

    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    # ==============================
    # Face orientation (eye line)
    # ==============================
    left_eye = P(33)
    right_eye = P(263)

    eye_center = (left_eye + right_eye) / 2
    dx, dy = right_eye - left_eye
    angle = np.degrees(np.arctan2(dy, dx))

    # ==============================
    # Face scale
    # ==============================
    left_face = P(127)
    right_face = P(356)
    chin = P(152)

    face_width = np.linalg.norm(right_face - left_face)
    face_height = np.linalg.norm(chin - eye_center)

    # ==============================
    # Fixed Forehead ROI (NO ADAPTATION)
    # ==============================
    roi_width = 0.50 * face_width
    roi_height = 0.28 * face_height

    center_x = eye_center[0]
    center_y = eye_center[1] - 0.40 * face_height

    rect = (
        (center_x, center_y),
        (roi_width, roi_height),
        angle
    )

    box = cv2.boxPoints(rect).astype(int)

    # ==============================
    # Mask ROI
    # ==============================
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, box, 255)

    roi = cv2.bitwise_and(frame, frame, mask=mask)

    # ==============================
    # Skin mask (YCrCb)
    # ==============================
    roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)

    skin_mask = cv2.inRange(roi_ycrcb, lower, upper)

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # ==============================
    # Skin ratio (THIS IS THE KEY)
    # ==============================
    skin_pixels = np.count_nonzero(skin_mask)
    roi_pixels = np.count_nonzero(mask)

    skin_ratio = skin_pixels / (roi_pixels + 1e-6)

    # ==============================
    # Alert logic (HONEST)
    # ==============================
    if skin_ratio < SKIN_ALERT_THRESHOLD:
        status_text = "Please move hair away from forehead"
        color = (0, 0, 255)   # RED
    else:
        status_text = "Forehead visible - ready to measure"
        color = (0, 255, 0)   # GREEN

    # ==============================
    # Visualization
    # ==============================
    cv2.drawContours(frame, [box], 0, color, 2)

    cv2.putText(
        frame,
        status_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Skin ratio: {skin_ratio:.2f}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow("Forehead ROI", frame)
    cv2.imshow("Skin Mask (debug)", skin_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
