# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:50:46 2020

@author: alanj
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

IMAGE_SIZE = [244,244]

train_path = r'C:\Users\alanj\Documents\Python\Practice\face_recognition\images\train/'
valid_path = r'C:\Users\alanj\Documents\Python\Practice\face_recognition\images\validation/'

def train_model():
# -----------------------Preprocessing----------------------------------------
    vgg = VGG16(input_shape = IMAGE_SIZE+[3], weights = 'imagenet', include_top = False)

    for layer in vgg.layers:
        layer.trainable = False

    folders = glob(r'C:\Users\alanj\Documents\Python\Practice\face_recognition\images\train/*')
    
    x = Flatten()(vgg.output)
    prediction = Dense(len(folders), activation = 'softmax')(x)

#-------------------Making the Model and Data---------------------------------

    model = Model(inputs=vgg.input,outputs=prediction)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory(r'C:\Users\alanj\Documents\Python\Practice\face_recognition\images\train/',
                                                     target_size = (244,244), batch_size = 5,
                                                     class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(r'C:\Users\alanj\Documents\Python\Practice\face_recognition\images\validation/',
                                                     target_size = (244,244), batch_size = 5,
                                                     class_mode = 'categorical')

    r = model.fit(
        training_set, validation_data = test_set, epochs = 2,
        steps_per_epoch = len(training_set), validation_steps = len(test_set),
        verbose=1
        )

    plt.plot(r.history['loss'], label = 'Training Loss')
    plt.plot(r.history['val_loss'], label = 'Validation Loss')
    plt.legend()
    plt.show()

    plt.savefig(r'C:\Users\alanj\Documents\Python\Practice\face_recognition\LossVal')
    
    plt.plot(r.history['accuracy'], label = 'Training Acc')
    plt.plot(r.history['val_accuracy'], label = 'Validation Acc')
    plt.legend()
    plt.show()

    plt.savefig(r'C:\Users\alanj\Documents\Python\Practice\face_recognition\ACCVal')

    model.save(r'C:\Users\alanj\Documents\Python\Practice\face_recognition\face_recognition.h5')
