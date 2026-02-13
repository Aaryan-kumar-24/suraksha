import cv2
import numpy as np
import time
from datetime import datetime

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

heatmap = np.zeros_like(prev_gray, dtype=np.float32)

activity_score = 0.0
alert_active = False
pulse_phase = 0

# ===== PARAMETERS =====
DECAY = 0.94
GAIN = 35
ALERT_THRESHOLD = 0.3

prev_time = time.time()

# ===== UI PADDING =====
LEFT_PAD = 40
TOP_PAD = 60
RIGHT_PAD = 260
LINE_GAP = 60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ===== FPS =====
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # ===== MOTION DETECTION =====
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion_strength = np.mean(diff)
    motion_area_ratio = np.sum(motion_mask > 0) / motion_mask.size
    activity = motion_strength * motion_area_ratio

    activity_score = 0.9 * activity_score + 0.1 * activity

    # ===== HEAT MAP =====
    heatmap = heatmap * DECAY + (motion_mask / 255.0) * GAIN
    heatmap = np.clip(heatmap, 0, 255)

    heat_display = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heat_display = heat_display.astype(np.uint8)

    red_layer = np.zeros_like(frame)
    red_layer[:, :, 2] = heat_display

    blended = cv2.addWeighted(frame, 1.0, red_layer, 0.6, 0)

    h, w = blended.shape[:2]

    # ===== STATUS =====
    status = "ACTIVE" if activity_score > ALERT_THRESHOLD else "CALM"
    color = (0, 0, 255) if status == "ACTIVE" else (0, 255, 0)

    time_str = datetime.now().strftime("%H:%M:%S")

    # ===== LARGE TEXT SETTINGS =====
    MAIN_SCALE = 1.1
    SMALL_SCALE = 1.0
    THICK = 3

    # ===== LEFT PANEL =====
    y = TOP_PAD
    cv2.putText(blended, f"Activity Score : {activity_score:.2f}",
                (LEFT_PAD, y), cv2.FONT_HERSHEY_SIMPLEX, MAIN_SCALE, (255, 255, 255), THICK)

    y += LINE_GAP
    cv2.putText(blended, f"Motion Strength: {motion_strength:.2f}",
                (LEFT_PAD, y), cv2.FONT_HERSHEY_SIMPLEX, SMALL_SCALE, (230, 230, 230), THICK)

    y += LINE_GAP
    cv2.putText(blended, f"Motion Area    : {motion_area_ratio*100:.1f}%",
                (LEFT_PAD, y), cv2.FONT_HERSHEY_SIMPLEX, SMALL_SCALE, (230, 230, 230), THICK)

    y += LINE_GAP
    cv2.putText(blended, f"Heat Level     : {np.mean(heatmap):.2f}",
                (LEFT_PAD, y), cv2.FONT_HERSHEY_SIMPLEX, SMALL_SCALE, (230, 230, 230), THICK)

    # ===== RIGHT PANEL =====
    x_right = w - RIGHT_PAD

    # FPS
    cv2.putText(blended, f"FPS : {fps:.1f}",
                (x_right, TOP_PAD),
                cv2.FONT_HERSHEY_SIMPLEX, SMALL_SCALE, (255, 255, 255), THICK)

    # ===== TIME WITH PADDING BOX =====
    time_text = f"Time : {time_str}"
    (text_w, text_h), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, SMALL_SCALE, THICK)

    pad_x, pad_y = 14, 12

    box_x1 = w - text_w - pad_x*2 - 40
    box_y1 = TOP_PAD + LINE_GAP - text_h - pad_y
    box_x2 = w - 40
    box_y2 = TOP_PAD + LINE_GAP + pad_y

    overlay = blended.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    blended = cv2.addWeighted(overlay, 0.35, blended, 0.65, 0)

    cv2.rectangle(blended, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 1)

    text_x = box_x1 + pad_x
    text_y = box_y2 - pad_y
    cv2.putText(blended, time_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, SMALL_SCALE, (255, 255, 255), THICK)

    # STATUS
    cv2.putText(blended, f"Status: {status}",
                (x_right - 30, TOP_PAD + 2 * LINE_GAP),
                cv2.FONT_HERSHEY_SIMPLEX, MAIN_SCALE, color, THICK)

    # ===== ALERT LATCH =====
    if activity_score > ALERT_THRESHOLD:
        alert_active = True
    if activity_score < 0.1:
        alert_active = False

    # ===== PROFESSIONAL ALERT =====
    if alert_active:
        pulse_phase += 0.2
        glow = int((np.sin(pulse_phase) + 1) * 40)

        box_width = 420
        box_height = 80
        x1 = (w - box_width) // 2
        y1 = h - box_height - 30
        x2 = x1 + box_width
        y2 = y1 + box_height

        alert_color = (0, 0, 180 + glow)
        cv2.rectangle(blended, (x1, y1), (x2, y2), alert_color, -1)
        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 255), 3)

        alert_text = " DANGER   DETECTED"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]

        text_x = x1 + (box_width - text_size[0]) // 2
        text_y = y1 + (box_height + text_size[1]) // 2 - 8

        cv2.putText(blended, alert_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    # ===== SHOW WINDOW =====
    cv2.imshow("Smart Motion Heat Camera", blended)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()