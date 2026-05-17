import cv2
import os

def check_cameras():
    results = []
    results.append("Checking available cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            results.append(f"Camera index {i} is available with CAP_DSHOW")
            cap.release()
        else:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                results.append(f"Camera index {i} is available with Default Backend")
                cap.release()
            else:
                results.append(f"Camera index {i} is NOT available")
    
    with open("camera_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    check_cameras()
