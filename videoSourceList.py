import cv2

def list_local_video_sources(max_tested=10):
    available_sources = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:  # If frame is read successfully
            available_sources.append(i)
        cap.release()
    return available_sources

sources = list_local_video_sources()
print("Available local video sources:", sources)
