from ultralytics import YOLO
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "test_on_videos"/ "test" / "test3.mp4"
OUTPUT_PATH = PROJECT_ROOT / "test_on_videos"/ "result" / "street_detected_ny.mp4"

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

    cv2.imshow("YOLO26n - Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved result to: {OUTPUT_PATH}")