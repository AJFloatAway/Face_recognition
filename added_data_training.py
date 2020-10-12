# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:47:07 2020

@author: alanj
"""


from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
from os.path import join
import shutil

tempTrainPath = r'.\images\train\temp/'
tempValidPath = r'.\images\validation\temp/'

def train_added_data():    
    model = load_model(r'.\face_recognition.h5')
    
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
    
    valid_datagen = ImageDataGenerator(rescale = 1./255)
    
    train_dataset = train_datagen.flow_from_directory(
        tempTrainPath,
        target_size = (244,244), 
        batch_size = 5,
        class_mode = 'categorical')
    valid_dataset = valid_datagen.flow_from_directory(
        tempValidPath,
        target_size = (244,244), 
        batch_size = 5,
        class_mode = 'categorical')
    
    model.fit(train_dataset, validation_data=valid_dataset, epochs=3, 
              verbose=1, steps_per_epoch=len(train_dataset), 
              validation_steps=len(valid_dataset))
    
    model.save(r'.\face_recognition.h5')
    
def move_data(name):
    trainPath = r'.\images\train/' + name
    validPath = r'.\images\validation/' + name
    
    for file in os.listdir(join(tempTrainPath, name)):
        shutil.move(tempTrainPath+file, trainPath)
    
    for file in os.listdir(join(tempValidPath, name)):
        shutil.move(tempValidPath+file, validPath)
        
    os.remove(tempTrainPath + name)
    os.remove(tempValidPath + name)