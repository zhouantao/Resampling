#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:26:24 2018

@author: dell
"""



from __future__ import print_function, division
import scipy

import keras
from keras.datasets import mnist
from keras.layers.wrappers import Bidirectional

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda,Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,MaxPooling2D,concatenate,GlobalAveragePooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,Adamax,SGD,Adadelta
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from keras.callbacks import TensorBoard
from keras.models import load_model

import matplotlib.pyplot as plt

from data_loader import load_data
import numpy as np

import cv2
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
import net
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping,TensorBoard
from keras.callbacks import ReduceLROnPlateau

class EfGan():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)

        self.train,self.val,self.test = load_data()
        
        
        self.optimizer = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)

        self.model = net.CreatModel()
        
        self.model.compile(loss='binary_crossentropy',optimizer = self.optimizer,metrics = ['accuracy'])
        #categorical_crossentropy
        
    def load_batch(self,batch_size = 1,data = None,classes = 4,batches = 10):          
        
            
        while(1):
            np.random.shuffle(data)
            for i in range(batches):
                batch_data = data[i*batch_size:(i+1) * batch_size,:]
                image,label = [],[]
                for image_list in batch_data:
                    img = self.read_image(image_list[0])                
                    image.append(img)
                    label.append(int(float(image_list[1])))
                    
                    
                   
                image = np.array(image)/255.0


                yield image,label
    
            
    def read_image(self,path):
        image = cv2.imread(path)
#        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = image[:,:,1]
        image = np.expand_dims(image,2)
        return image 
    

    def TrainModel(self):
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=3, verbose=1)

        csv_log = CSVLogger('my128up.csv')
        callbacks = [reduce_lr,csv_log]
        
        
        self.model.fit_generator(generator = self.load_batch(batch_size = 32,data = self.train,classes = 3,batches= 1562), 
                                 steps_per_epoch = 1562, epochs = 30,callbacks =callbacks,
                                 validation_data = self.load_batch(batch_size = 25,data = self.val,classes = 3,batches = 400),
                                 validation_steps=400)
        
        x = self.model.evaluate_generator(self.load_batch(batch_size = 25,data = self.test,classes = 3,batches = 400),400)
        print(x)
        self.model.save('downscale50_jpeg70.h5')
            
model =  EfGan()
#model.model.summary() 
#print('Train:downscale50_jpeg70')    
model.TrainModel()    
#print('Train:downscale50_jpeg70')  
    

    
        
