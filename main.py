from deepface import DeepFace
import cv2 
import pandas as pd
import os

#pre-trained models
#compatible python3.9, deepface
#https://github.com/serengil/deepface_models/releases/
#to C:\Users\phalla\.deepface\weights\

'''
#For testing
img = cv2.imread("faces/alex.jpg")
results = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion")) #race, emotion
print(results)
'''

data= {
    "Name": [],
    "Age": [],
    "Gender": [],
    "Race": []
}

for file in os.listdir("faces"):
    result = DeepFace.analyze(cv2.imread(f"faces/{file}"), actions=("age", "gender", "race"))
    data["Name"].append(file.split(".")[0])
    data["Age"].append(result[0]["age"])
    data["Gender"].append(result[0]["dominant_gender"])
    data["Race"].append(result[0]["dominant_race"])

df = pd.DataFrame(data)
print(df)

df.to_csv("people.csv")