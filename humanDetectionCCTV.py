import cv2
import os
import time
import threading
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # Or yolov8m.pt or yolov8l.pt if needed
save_interval = 2  # Seconds between image saves if person is detected

# Output directory
output_folder = "cctv_detections"
os.makedirs(output_folder, exist_ok=True)

# List of your CCTV RTSP URLs (replace with actual values)
camera_sources = [
    "rtsp://admin:password@192.168.1.101:554/stream1",
    "rtsp://admin:password@192.168.1.102:554/stream1",
    "rtsp://admin:password@192.168.1.103:554/stream1",
    "rtsp://admin:password@192.168.1.104:554/stream1"
]

def run_detection(camera_url, cam_name):
    cap = cv2.VideoCapture(camera_url)
    last_saved_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] Failed to get frame from {cam_name}")
            time.sleep(2)
            continue

        frame_resized = cv2.resize(frame, (640, 480))
        results = model(frame_resized, verbose=False)[0]

        # Check if any 'person' is detected
        person_detected = any(model.names[int(box.cls[0])] == 'person' for box in results.boxes)

        # Save snapshot if detected
        current_time = time.time()
        if person_detected and (current_time - last_saved_time >= save_interval):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{cam_name}_{timestamp}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame_resized)
            print(f"[{cam_name}] Person detected. Saved: {filename}")
            last_saved_time = current_time

        # Optional: show live feed (comment out if running headless)
        cv2.imshow(cam_name, frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start a thread for each CCTV camera
for idx, cam_url in enumerate(camera_sources):
    cam_name = f"Camera_{idx+1}"
    t = threading.Thread(target=run_detection, args=(cam_url, cam_name), daemon=True)
    t.start()

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("[INFO] Stopping detection.")
