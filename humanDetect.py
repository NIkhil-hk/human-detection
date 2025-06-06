import cv2
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
import imghdr
import time

# === Configuration ===
EMAIL_SENDER = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_app_password'  # Use app password if 2FA is on
EMAIL_RECEIVER = 'your_email@gmail.com'

# CCTV feed (0 for webcam, or use RTSP/HTTP URL)
VIDEO_SOURCE = 0  # Change to your CCTV IP camera URL if needed

# Create output folder
output_folder = "detections"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set up the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def send_email(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Human Detected by CCTV'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content('A person was detected by the CCTV. See the attached image.')

    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        img_type = imghdr.what(img_file.name)
        img_name = os.path.basename(img_file.name)

    msg.add_attachment(img_data, maintype='image', subtype=img_type, filename=img_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

def detect_and_alert():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    last_sent = 0
    cool_down = 30  # seconds before next email

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect people
        boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))

        if len(boxes) > 0:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{output_folder}/Human_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Human detected. Snapshot saved: {filename}")

            # Send email if cooldown passed
            if time.time() - last_sent > cool_down:
                send_email(filename)
                print("[INFO] Email sent with snapshot.")
                last_sent = time.time()
            else:
                print("[INFO] Email not sent due to cooldown.")

        # Optional: display for debugging
        # cv2.imshow("CCTV Feed", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_alert()
