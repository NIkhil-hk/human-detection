import cv2
import os
from datetime import datetime
from ultralytics import YOLO
import time

# === Load Model ===
body_model = YOLO("yolov8s.pt")  # Use yolov8s.pt for better accuracy 
# we can also use the yolov8n for better speed but this mode will have less accuracy as compared to that of the small model


# === Create output folder ===
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)

# function to list all the connected cameras

# === Open webcam or CCTV stream ===
cap = cv2.VideoCapture(0)  # Use stream URL instead of 0 if needed


print("[INFO] Starting detection. Press 'q' to quit.")

last_saved_time = 0
save_interval = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    frame_resized = cv2.resize(frame, (640, 480))

    # === Run YOLOv8 for person detection ===
    results = body_model(frame_resized, verbose=False)[0]
    body_detected = False

    for box in results.boxes:
        cls = int(box.cls[0])
        label = body_model.names[cls]
        if label == "person":
            body_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    current_time = time.time()

    # === Save snapshot every 5 seconds if person detected ===
    if body_detected and (current_time - last_saved_time) >= save_interval:
        now = datetime.now()
        filename_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_path = os.path.join(output_folder, f"pic_{filename_time}.jpg")
        cv2.imwrite(snapshot_path, frame_resized)
        print(f"[SAVED] Human detected. Snapshot saved: {snapshot_path}")
        last_saved_time = current_time

    # === Display the video feed ===
    cv2.imshow("Human Detection", frame_resized)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
