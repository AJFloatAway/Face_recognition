# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:16:43 2020

@author: alanj
"""

import cv2
import csv
import operator
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(r'.\haarcascade_frontalface_default.xml')




def data_collection(name, age, phone):
    
    def face_extractor(img):
        faces = face_classifier.detectMultiScale(img, 1.3, 5)
        if faces is():
            return None
    
        for(x,y,w,h) in faces:
            x = x-10
            y = y-10
            cropped_face = img[y:y+h+50, x:x+w+50]
        
        return cropped_face

    profile = open(r'./Profiles.csv', 'r')
    csv1 = csv.reader(profile,delimiter=',')
    sort = sorted(csv1, key=operator.itemgetter(0))
    
    video = cv2.VideoCapture(1)
    video.set(cv2.CAP_PROP_FPS, 60)
    count = 0

    ageFile = r'.\profile_data/' + name + '/' + 'age.txt'
    phoneFile = r'.\profile_data/' + name + '/' + 'phone.txt'
    
    # write age and phone to files
    
    with open(ageFile, "w") as file:
        file.write(age)
    with open(phoneFile, "w") as file:
        file.write(phone)

    while True:
        check, frame = video.read()    
    
        if face_extractor(frame) is not None:
            count +=1
            face = cv2.resize(face_extractor(frame), (200,200))
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            file_name_path = r'.\images\validation/' + name + '/' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
            print(plt.imread(file_name_path))
            imgplot = plt.imshow(plt.imread(file_name_path))
            plt.show(imgplot)
            
            cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face Not Found")
            pass
    
        if cv2.waitKey(1) & 0xFF == ord('q') or count == 10:
            count = 0
            break
        
    while True:
        check, frame = video.read()    
    
        if face_extractor(frame) is not None:
            count +=1
            face = cv2.resize(face_extractor(frame), (400,400))
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            file_name_path = r'.\images\train/' + name + '/' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
            print(plt.imread(file_name_path))
            imgplot = plt.imshow(plt.imread(file_name_path))
            plt.show(imgplot)
            
            cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face Not Found")
            pass
    
        if cv2.waitKey(1) & 0xFF == ord('q') or count == 30:
            break

    video.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete")
    imgfilename = r'.\images\validation/' + name + '/1.jpg'
    
    