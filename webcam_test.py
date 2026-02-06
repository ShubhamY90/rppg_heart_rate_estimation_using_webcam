import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam cannot be opened")
    exit()

print("Webcam opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Frame not received")
        break

    cv2.imshow("Webcam Test - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
