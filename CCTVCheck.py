import cv2

def check_cctv_connection(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    cap.release()

    if ret:
        print("[✅ CONNECTED] CCTV stream is active.")
        return True
    else:
        print("[❌ NOT CONNECTED] Could not read from CCTV.")
        return False

# Example usage
rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream1"
check_cctv_connection(rtsp_url)


#nmap -p 554 --open 192.168.1.0/24
