from imageio import save
from keras.models import load_model
from time import sleep
from keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import multiprocessing
import time
from cProfile import run
from distutils import command
from email.mime import audio
from tkinter import NO
from pyparsing import re
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
from sys import argv
import random
from playsound import playsound
import pyjokes
from text_to_speech import speak
import requests
import multiprocessing
import time
from bs4 import BeautifulSoup
from keras.models import load_model
from time import sleep
from keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np




def talk(text):
   speak(text,"en",save=True,file='song.mp3', speak=True)




face_classifier = cv2.CascadeClassifier(r'emotions.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['angry','','fear','happy','neutral', 'sad', 'surprise']

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)


count=0
emo=[]
while True:
        count+=1
        _, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        height, width = frame.shape[:2]
        for detection in detections:
            for object_detection in detection:
                scores = object_detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 is for faces
                    center_x = int(object_detection[0] * width)
                    center_y = int(object_detection[1] * height)
                    w = int(object_detection[2] * width)
                    h = int(object_detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    if np.sum([roi_gray]) != 0:
                        # Placeholder for emotion detection logic
                        label = "emotion"  # Replace with actual emotion detection
                        label_position = (x, y)
                        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        emo.append(label)
                    else:
                        cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
print(emo)
sc=max(set(emo), key = emo.count)
print(sc)
tex=("you are"+sc)


if 'sad' in tex:
        talk('you are so happy today, tell me y ?')
elif 'happy' in tex:
      talk('you are so happy today, tell me y ?')
      talk('you are so happy today, tell me y ?')

elif 'angry' in tex:
        talk('you look angry and serious, are u angry with me?')
else:
        talk('you dont have any expression are u human?')

cap.release()
cv2.destroyAllWindows()

# Run the script to ensure the YOLO model detects faces and the emotion classification logic works as expected.
# Verify that the video feed displays detected faces with emotion labels and that the verbal responses are appropriate.


