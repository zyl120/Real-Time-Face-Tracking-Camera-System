import cv2
import numpy as np
import os
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
Face_Images = os.path.join(os.getcwd(), "Face_Images")

Face_ID = 0
pev_person_name = None
x_train = []
y_ID = []
num_photos = 0
for root, dirs, files in os.walk(Face_Images):
    for file in files:  # check every directory in it
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            person_name = os.path.basename(root)

            num_photos += 1
            print("Path: {}".format(path))
            print("Name: {}".format(person_name))
            print("FaceID: {}".format(Face_ID))
            

            if pev_person_name != person_name:
                Face_ID = Face_ID + 1  # If yes, increment the ID count
                pev_person_name = person_name

            Grey_Image = Image.open(path).convert("L")
            Crop_Image = Grey_Image.resize((550, 550), Image.ANTIALIAS)
            Final_Image = np.array(Crop_Image, "uint8")
            faces = face_cascade.detectMultiScale(Final_Image, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = Final_Image[y:y+h, x:x+w]
                x_train.append(roi)
                y_ID.append(Face_ID)

recognizer.train(x_train, np.array(y_ID))
recognizer.save("face-trainner.yml")
print("num_photos: {}".format(num_photos))
