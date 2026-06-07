import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
url = "rtsp://street:12345678@192.168.1.50:554/stream1"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Could not open stream")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("stream1", frame)
    if cv2.waitKey(1) == 27:   # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()