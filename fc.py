import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================
# PARAMETERS
# ==============================
SKIN_ALERT_THRESHOLD = 0.60   # below this → insufficient exposed skin
GLASSES_THRESHOLD = 0.35      # if more than 15% non-skin in eye region → glasses detected

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
        cv2.imshow("Face ROI", frame)
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
    nose_tip = P(1)

    face_width = np.linalg.norm(right_face - left_face)
    face_height = np.linalg.norm(chin - eye_center)

    # ==============================
    # IMPROVED FOREHEAD ROI (tighter, more adaptive)
    # ==============================
    forehead_top = P(10)  # Top of forehead landmark
    
    forehead_width = 0.45 * face_width  # Slightly narrower
    forehead_height = 0.25 * face_height  # Slightly shorter

    forehead_x = eye_center[0]
    forehead_y = eye_center[1] - 0.38 * face_height  # Slightly lower

    forehead_rect = (
        (forehead_x, forehead_y),
        (forehead_width, forehead_height),
        angle
    )
    forehead_box = cv2.boxPoints(forehead_rect).astype(int)

    # ==============================
    # LEFT CHEEK ROI (using actual cheek landmarks)
    # ==============================
    left_cheekbone = P(266)  # Left cheekbone
    left_jaw = P(323)         # Left jaw point
    
    left_cheek_width = 0.28 * face_width
    left_cheek_height = 0.25 * face_height

    # Position between nose and left face edge
    left_cheek_x = eye_center[0] - 0.30 * face_width
    left_cheek_y = nose_tip[1] - 0.10 * face_height  # Below nose level

    left_cheek_rect = (
        (left_cheek_x, left_cheek_y),
        (left_cheek_width, left_cheek_height),
        angle
    )
    left_cheek_box = cv2.boxPoints(left_cheek_rect).astype(int)

    # ==============================
    # RIGHT CHEEK ROI (using actual cheek landmarks)
    # ==============================
    right_cheekbone = P(36)   # Right cheekbone
    right_jaw = P(93)          # Right jaw point
    
    right_cheek_width = 0.28 * face_width
    right_cheek_height = 0.25 * face_height

    # Position between nose and right face edge
    right_cheek_x = eye_center[0] + 0.30 * face_width
    right_cheek_y = nose_tip[1] - 0.10 * face_height  # Below nose level

    right_cheek_rect = (
        (right_cheek_x, right_cheek_y),
        (right_cheek_width, right_cheek_height),
        angle
    )
    right_cheek_box = cv2.boxPoints(right_cheek_rect).astype(int)

    # ==============================
    # GLASSES DETECTION (check area around eyes)
    # ==============================
    # Create mask for eye region
    left_eye_points = np.array([
        P(33), P(160), P(159), P(158), P(157), P(173),
        P(133), P(155), P(154), P(153), P(145), P(144)
    ], dtype=np.int32)
    
    right_eye_points = np.array([
        P(263), P(387), P(386), P(385), P(384), P(398),
        P(362), P(382), P(381), P(380), P(374), P(373)
    ], dtype=np.int32)

    eye_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(eye_mask, left_eye_points, 255)
    cv2.fillConvexPoly(eye_mask, right_eye_points, 255)
    
    # Expand eye mask to include glasses frame area
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eye_mask = cv2.dilate(eye_mask, kernel_dilate)

    # Get eye region
    eye_roi = cv2.bitwise_and(frame, frame, mask=eye_mask)
    eye_ycrcb = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2YCrCb)
    
    # Detect non-skin in eye area (likely glasses)
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    
    eye_skin_mask = cv2.inRange(eye_ycrcb, lower_skin, upper_skin)
    
    eye_total_pixels = np.count_nonzero(eye_mask)
    eye_skin_pixels = np.count_nonzero(eye_skin_mask)
    eye_non_skin_ratio = 1.0 - (eye_skin_pixels / (eye_total_pixels + 1e-6))
    
    glasses_detected = eye_non_skin_ratio > GLASSES_THRESHOLD

    # ==============================
    # Combined mask for all ROIs
    # ==============================
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(combined_mask, forehead_box, 255)
    cv2.fillConvexPoly(combined_mask, left_cheek_box, 255)
    cv2.fillConvexPoly(combined_mask, right_cheek_box, 255)

    roi = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # ==============================
    # IMPROVED Skin Detection (HSV + YCrCb hybrid)
    # ==============================
    roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # YCrCb skin detection
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask_ycrcb = cv2.inRange(roi_ycrcb, lower_ycrcb, upper_ycrcb)

    # HSV skin detection (helps with hair)
    lower_hsv = np.array([0, 15, 0], dtype=np.uint8)
    upper_hsv = np.array([25, 170, 255], dtype=np.uint8)
    skin_mask_hsv = cv2.inRange(roi_hsv, lower_hsv, upper_hsv)

    # Combine both masks
    skin_mask = cv2.bitwise_and(skin_mask_ycrcb, skin_mask_hsv)

    # More aggressive cleaning for hair detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ==============================
    # Individual ROI analysis
    # ==============================
    forehead_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(forehead_mask, forehead_box, 255)
    
    forehead_skin = cv2.bitwise_and(skin_mask, forehead_mask)
    forehead_skin_pixels = np.count_nonzero(forehead_skin)
    forehead_total_pixels = np.count_nonzero(forehead_mask)
    forehead_ratio = forehead_skin_pixels / (forehead_total_pixels + 1e-6)

    # ==============================
    # Overall skin ratio
    # ==============================
    skin_pixels = np.count_nonzero(skin_mask)
    roi_pixels = np.count_nonzero(combined_mask)

    skin_ratio = skin_pixels / (roi_pixels + 1e-6)

    # ==============================
    # Alert logic
    # ==============================
    alerts = []
    
    if forehead_ratio < SKIN_ALERT_THRESHOLD:
        alerts.append("Move hair from forehead")
    
    if skin_ratio < SKIN_ALERT_THRESHOLD:
        alerts.append("Ensure face is clearly visible")
    
    if glasses_detected:
        alerts.append("Glasses detected - may affect accuracy")
    
    if not alerts:
        status_text = "Ready to measure"
        color = (0, 255, 0)   # GREEN
    else:
        status_text = " | ".join(alerts)
        color = (0, 0, 255)   # RED

    # ==============================
    # Visualization
    # ==============================
    # Draw ROI boxes
    cv2.drawContours(frame, [forehead_box], 0, color, 2)
    cv2.drawContours(frame, [left_cheek_box], 0, color, 2)
    cv2.drawContours(frame, [right_cheek_box], 0, color, 2)

    # Labels
    cv2.putText(frame, "F", tuple(forehead_box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, "L", tuple(left_cheek_box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, "R", tuple(right_cheek_box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Status
    cv2.putText(
        frame,
        status_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Metrics
    y_offset = 70
    cv2.putText(frame, f"Overall: {skin_ratio:.2f}", (30, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_offset += 30
    cv2.putText(frame, f"Forehead: {forehead_ratio:.2f}", (30, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_offset += 30
    
    if glasses_detected:
        cv2.putText(frame, "GLASSES: YES", (30, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("Face ROI", frame)
    cv2.imshow("Skin Mask (debug)", skin_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()