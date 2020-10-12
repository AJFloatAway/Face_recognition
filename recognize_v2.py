# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:24:15 2020

@author: alanj
"""
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import tkinter as tk
import os
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(3)

os.system('cls')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main_app():
    face_classifier = cv2.CascadeClassifier(r'.\haarcascade_frontalface_default.xml')
    model = load_model(r'.\face_recognition.h5')
    
    video = cv2.VideoCapture(1)
    cv2.namedWindow("Window")
    
    while True:
        _,frame = video.read()
        faces = face_classifier.detectMultiScale(frame, 1.3, 5)
        if faces is():
            cropped_face = None
        else:
            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255),2)
                x = x-10
                y = y-10
                cropped_face = frame[y:y+h+50, x:x+w+50]
        
        face = cropped_face
        if type(face) is np.ndarray:
            face = cv2.resize(face,(224,244))
            im = Image.fromarray(face,'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis = 0)
            pred = model.predict(img_array)
    
            name = 'None Matching'
            
            print(pred)
            pred.class_indices
            
            for i in pred[0]:
                if (pred[0][i] > 0.6):
                    name = pred[1][i]
                    cv2.putText(frame, name, (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)  
        else:
            cv2.putText(frame, "NO FACE FOUND", (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        ''' if(pred[0][0]>0.9):
                
                name = 'Alan'
                cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)  
            elif(pred[0][1]>0.9):
                name = 'Ming'
                cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            elif(pred[0][2]>0.9):
                name = 'Richard'
                cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            elif(pred[0][3]>0.9):
                name = 'Simon'
                cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
            elif(pred[0][4]>0.9):
                name = 'William'
                cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            else:
                cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)  
        '''
        
    
        cv2.imshow('Window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()
