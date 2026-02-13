import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Auto-create paths
path = 'ImagesAttendance'
os.makedirs(path, exist_ok=True)

attendance_file = 'Attendance.csv'
if not os.path.exists(attendance_file):
    open(attendance_file, 'w').close()

images = []
classNames = []

# Load registered faces
for cl in os.listdir(path):
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

print("Registered faces:", classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList


# 🔹 Update latest time instead of skipping duplicate
def markAttendance(name):
    now = datetime.now()
    dt = now.strftime('%H:%M:%S')

    lines = []
    updated = False

    with open(attendance_file, 'r') as f:
        lines = f.readlines()

    with open(attendance_file, 'w') as f:
        for line in lines:
            entry_name = line.split(',')[0]

            if entry_name == name:
                f.write(f"{name},{dt}\n")  # update time
                updated = True
            else:
                f.write(line)

        if not updated:
            f.write(f"{name},{dt}\n")  # new entry


encodeListKnown = findEncodings(images)
print("Encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(imgSmall)
    encodes = face_recognition.face_encodings(imgSmall, faces)

    for encodeFace, faceLoc in zip(encodes, faces):

        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = "UNKNOWN"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0,200), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0,225), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 2)

    cv2.imshow("Attendance System", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()