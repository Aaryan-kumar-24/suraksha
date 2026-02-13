import cv2
import pickle
import numpy as np

# Webcam
cap = cv2.VideoCapture(0)

# Load saved box positions
with open('box2', 'rb') as f:
    posList = pickle.load(f)


def checkTouch(imgPro, img):
    video_h, video_w = img.shape[:2]
    alert = False

    for pos in posList:
        x_rel, y_rel, w_rel, h_rel = pos

        # Convert relative → pixel
        x = int(x_rel * video_w)
        y = int(y_rel * video_h)
        w = int(w_rel * video_w)
        h = int(h_rel * video_h)

        imgCrop = imgPro[y:y + h, x:x + w]
        count = cv2.countNonZero(imgCrop)

        # 🔴 If any motion detected → ALERT immediately
        if count > 1:
            color = (0, 0, 255)  # Red
            thickness = 3
            alert = True
        else:
            color = (0, 255, 0)  # Green
            thickness = 2

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    return alert


while True:
    success, img = cap.read()
    if not success:
        continue

    # Pre-processing for motion detection
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 16
    )
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    alert = checkTouch(imgDilate, img)

    # 🔴 Show ALERT text instantly
    if alert:
        cv2.putText(img, "ALERT!",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

    cv2.imshow("Touch Detection (Press A to Exit)", img)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
