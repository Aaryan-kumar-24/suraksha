import cv2
import pickle
import numpy as np
import time

# Use laptop webcam instead of video
cap = cv2.VideoCapture(0)

# Load saved  box positions
with open('box', 'rb') as f:
    posList = pickle.load(f)

# Dictionary to track occupied time
occupied_start_time = {}


def checkParkingSpace(imgPro, img):
    video_h, video_w = img.shape[:2]
    alert = False

    for idx, pos in enumerate(posList):
        x_rel, y_rel, w_rel, h_rel = pos

        # Convert relative → pixel
        x = int(x_rel * video_w)
        y = int(y_rel * video_h)
        w = int(w_rel * video_w)
        h = int(h_rel * video_h)

        imgCrop = imgPro[y:y + h, x:x + w]
        count = cv2.countNonZero(imgCrop)
        if count < 2:
            color = (0, 255, 0)  # Green = free
            thickness = 2
            occupied_start_time.pop(idx, None)

        else:
            if idx not in occupied_start_time:
                occupied_start_time[idx] = time.time()

            elapsed = time.time() - occupied_start_time[idx]

            if elapsed > 2:
                color = (0, 0, 255)  # Red = long wait
                thickness = 3
                cv2.putText(img, "Waiting too long!", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                alert = True
            else:
                color = (0, 165, 255)  # Orange = short wait
                thickness = 2

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    return alert


while True:
    success, img = cap.read()
    if not success:
        continue

    # Preprocessing
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 16
    )
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    alert = checkParkingSpace(imgDilate, img)

    if alert:
        cv2.putText(img, "ALERT!", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("(Press A to Exit)", img)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()