import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load Face Landmarker model
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # Forehead landmarks (indices work here too)
        ids = [10, 338, 297, 332]
        xs = [int(lm[i].x * w) for i in ids]
        ys = [int(lm[i].y * h) for i in ids]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Live ROI (MediaPipe Tasks)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
