from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
model = YOLO(MODEL_PATH)
print(model.names)