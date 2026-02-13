import cv2
import os

# Auto-create folder
save_path = "ImagesAttendance"
os.makedirs(save_path, exist_ok=True)

# 🔹 Ask name FIRST
first_name = input("Enter FIRST NAME to register: ").strip().lower()

if first_name == "":
    print("❌ Name cannot be empty!")
    exit()

file_path = os.path.join(save_path, f"{first_name}.jpg")

cap = cv2.VideoCapture(0)

print("Press 's' to capture face | 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    cv2.imshow("Register Face", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite(file_path, img)
        print(f"✅ Face registered for {first_name}")
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()