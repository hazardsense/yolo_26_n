from ultralytics import YOLO
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
IMAGE_PATH = PROJECT_ROOT / "test_on_video" / "photo"/ "test11.jpg"

# Load model
model = YOLO(MODEL_PATH)

# Run detection
results = model(IMAGE_PATH, conf=0.5)

# Get annotated image
annotated_image = results[0].plot()

# Display
cv2.imshow("YOLO26n Detection", annotated_image)

print("\nDetected objects:")

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    print(f"{class_name}: {confidence:.2f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
