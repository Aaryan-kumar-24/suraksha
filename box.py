import cv2
import pickle
import os

# 🔴 Delete old CarParkPos file on start
if os.path.exists("box"):
    os.remove("box")

# Start with empty list
posList = []

cap = cv2.VideoCapture(0)

# Fixed rectangle size
box_w, box_h = 200, 200


def mouseClick(event, x, y, flags, params):
    global posList, img_shape

    if event == cv2.EVENT_LBUTTONDOWN:
        img_h, img_w = img_shape[:2]

        # 🔹 Check if clicked inside existing box → DELETE
        for i, pos in enumerate(posList):
            x1, y1, w1, h1 = pos
            abs_x = int(x1 * img_w)
            abs_y = int(y1 * img_h)
            abs_w = int(w1 * img_w)
            abs_h = int(h1 * img_h)

            if abs_x < x < abs_x + abs_w and abs_y < y < abs_y + abs_h:
                posList.pop(i)
                with open('box', 'wb') as f:
                    pickle.dump(posList, f)
                return

        # 🔹 Otherwise → CREATE new box
        x_rel = x / img_w
        y_rel = y / img_h
        w_rel = box_w / img_w
        h_rel = box_h / img_h

        posList.append((x_rel, y_rel, w_rel, h_rel))

        with open('box', 'wb') as f:
            pickle.dump(posList, f)


cv2.namedWindow("Selector")
cv2.setMouseCallback("Selector", mouseClick)

while True:
    success, img = cap.read()
    if not success:
        continue

    img_shape = img.shape

    # Draw rectangles
    for pos in posList:
        x_rel, y_rel, w_rel, h_rel = pos
        x = int(x_rel * img_shape[1])
        y = int(y_rel * img_shape[0])
        w = int(w_rel * img_shape[1])
        h = int(h_rel * img_shape[0])

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.putText(img, "Left Click: Create/Delete | A: Exit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Selector", img)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()