import cv2
import face_recognition

img = cv2.imread("people/Messi.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert format
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("people/Ronaldo.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) #convert format
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

img3 = cv2.imread("people/Moto.jpg")
rgb_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB) #convert format
img_encoding3 = face_recognition.face_encodings(rgb_img3)[0]

result = face_recognition.compare_face([img_encoding], img_encoding2)
print("Result: ", result)


cv2.imshow("Img", img)
cv2.imshow("Img 2", img2)
cv2.imshow("Img 3", img3)
cv2.waitKey(0)