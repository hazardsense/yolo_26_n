from ultralytics import YOLO
import cv2

MODEL_PATH = "../models/best.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO26n - ROD", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()