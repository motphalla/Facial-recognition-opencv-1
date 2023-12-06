#add rectangle line on the face and reposition the name to the right top of the rectangle line
#2023/12/04

import threading
import cv2
import os
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

reference_images = {}

names = ["phalla", "john", "akari"]
for name in names:
    img_path = os.path.join("faces", f"{name}.jpg")
    reference_images[name] = cv2.imread(img_path)

def check_face(frame):
    global face_match
    try:
        for name, reference_img in reference_images.items():
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                return name
        return None
    except ValueError:
        return None

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                result = check_face(frame.copy())
                if result is not None:
                    name = result
            except ValueError:
                pass
        counter += 1

        # Draw rectangle around the face
        if result is not None:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Reposition the name to the right top of the rectangle
                cv2.putText(frame, name, (x + w + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Recognized: {name}" if result else "No Match", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result else (0, 0, 255), 2)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
