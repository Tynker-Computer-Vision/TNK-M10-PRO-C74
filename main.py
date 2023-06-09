import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from cvzone.FaceDetectionModule import FaceDetector


path = "Face-image-dataset"
images = []
age = []
gender = []

detector = FaceDetector()

try:
  for img in os.listdir(path):
    print(img)
    if img!='.git':
      ages = img.split("_")[0]
      genders = img.split("_")[1]
      img = cv2.imread(str(path)+"/"+str(img))
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      
      # Crop the face out of an image and size the image at 200,200
      img, bbox = detector.findFaces(img, draw=False) 
      if bbox:
        X, Y, W, H = bbox[0]['bbox']
       
        croppedImg = img[Y:Y+H, X:X+W]
        resizedImg = [cv2.resize(croppedImg, (200, 200))]
                
        images.append(resizedImg)
        
        images.append(img)
        age.append(ages)
except Exception as e:
  print(e)

age = np.array(age,dtype=np.int64)
images = np.array(images)
try:
  print("age:", age)
  print("Images", images)
except Exception as e:
  print(e)
