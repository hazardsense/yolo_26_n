from ultralytics import YOLO

MODEL_PATH = "../models/best.pt"
model = YOLO(MODEL_PATH)
print(model.names)