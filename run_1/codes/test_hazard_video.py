import cv2
import time
import pyttsx3
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "test_on_videos"/ "test" / "test2.mp4"
OUTPUT_PATH = PROJECT_ROOT / "test_on_videos" / "result" / "street_detected_hazard.mp4"

# ---- CONFIG ----
# COCO class names relevant to pedestrian hazards (subset of the 80 pretrained classes)
HAZARD_CLASSES = {
   'Bike', 'Building','Car', 'Person', 'Stairs', 'Traffic sign', 'Electrical Pole', 
   'Road', 'Motorcycle', 'Dustbin', 'Dog', 'Manhole', 'Tree', 'Guard rail', 
   'Pedestrian crosswalk', 'Truck', 'Bus', 'Bench', 'Traffic Cone', 'Fire hydrant', 
   'Teraffic Barrel', 'Plant Pot', 'Electrical Box', 'Chair', 'Bicycle Rack' }

# How close (as a fraction of frame height) a box needs to be before we call it "close"
CLOSE_THRESHOLD = 0.5   # box height > 50% of frame height = treat as close/urgent
ALERT_COOLDOWN = 3.0    # seconds between repeat alerts for the same object, so it doesn't spam

# ---- SETUP ---
model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(str(VIDEO_PATH))
model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))
model = YOLO(str(MODEL_PATH))  # smallest/fastest pretrained model, good enough for a demo
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)

last_alert_time = {}  # tracks cooldown per object class

def speak(text):
    print(f"[ALERT] {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def get_direction(x_center, frame_width):
    """Rough left/center/right estimate based on horizontal position."""
    if x_center < frame_width * 0.33:
        return "on your left"
    elif x_center > frame_width * 0.66:
        return "on your right"
    else:
        return "ahead"

def main():
    if not VIDEO_PATH.is_file():
        print(f"Video file not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    output_path = OUTPUT_PATH
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        print(f"Could not create output video: {output_path}")
        cap.release()
        return

    print(f"Starting hazard detector with {VIDEO_PATH.name}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label not in HAZARD_CLASSES or conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height = y2 - y1
            closeness_ratio = box_height / frame_height
            x_center = (x1 + x2) / 2

            # Draw box for visual demo purposes
            color = (0, 0, 255) if closeness_ratio > CLOSE_THRESHOLD else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Only alert for close/urgent hazards, respecting cooldown
            if closeness_ratio > CLOSE_THRESHOLD:
                now = time.time()
                if now - last_alert_time.get(label, 0) > ALERT_COOLDOWN:
                    direction = get_direction(x_center, frame_width)
                    speak(f"{label} {direction}")
                    last_alert_time[label] = now

        writer.write(frame)
        cv2.imshow("Hazard Detector Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved detection video to: {output_path}")

if __name__ == "__main__":
    main()