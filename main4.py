import threading
import cv2
import os
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

# Dictionary to store reference images for each person
reference_images = {}

# Load reference images for each person
names = ["phalla", "john", "angela"]  # Add more names as needed
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
                    print(f"Face recognized: {name}")
            except ValueError:
                pass
        counter += 1

        cv2.putText(frame, f"Recognized: {name}" if result else "No Match", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result else (0, 0, 255), 2)

        cv2.imshow("video", frame)

    # press q to break the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
