#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:25:33 2018

@author: dell
"""
from __future__ import print_function, division


from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda,Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,MaxPooling2D,concatenate,GlobalAveragePooling2D,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,Adamax,SGD,Adadelta

from keras import initializers
from keras.callbacks import TensorBoard
from keras.models import load_model

import matplotlib.pyplot as plt

import numpy as np



def CreatModel():
    #91.14999
    def conv1(X):
        X = Conv2D(64,(5,5),strides = (1,1),padding = 'same')(X)
        X= BatchNormalization()(X)
        X = Activation('tanh')(X)        
        X = MaxPooling2D((3,3),strides = (2,2))(X)
        return X
    
    def conv2(X):
        X = Conv2D(64,(3,3),strides = (1,1),padding = 'same')(X)
        X= BatchNormalization()(X)
        X = Activation('tanh')(X)        
        X = MaxPooling2D((3,3),strides = (2,2))(X)
        return X
    
    def conv3(X):
        X = Conv2D(128,(3,3),strides = (1,1),padding = 'same')(X)
        X= BatchNormalization()(X)
        X = Activation('tanh')(X)        
        X = GlobalAveragePooling2D()(X)
        return X
    def conv1_1(X,counts):
        X = Conv2D(counts,(1,1),strides = (1,1),padding = 'same')(X)
        X= BatchNormalization()(X)
        X = Activation('tanh')(X)        
        return X
    
    
    input_shape = Input(shape=(64,64,1)) 

        
    pathway1 = Conv2D(1,(1,3),strides = (1,1),padding = 'same',trainable=False,use_bias = False,name = 'HPF1',
                      kernel_initializer = initializers.Constant(value=np.array([0.5,-1,0.5])))(input_shape)
    pathway2 = Conv2D(1,(3,1),strides = (1,1),padding = 'same',trainable=False,use_bias =False,name = 'HPF2',
                      kernel_initializer = initializers.Constant(value=np.array([[0.5],[-1],[0.5]])))(input_shape)
    
    X1_1 = conv1(pathway1)    
    X2_1 = conv1(pathway2)
    
    X1 = concatenate([X1_1, X2_1], axis=-1)
    X1 = conv1_1(X1,64)
    X2 = conv1(X1)
    X3 = conv1(X2)
    X4 = conv2(X3)
    
    X1_2 = conv2(X1_1)
    X2_2 = conv2(X2_1)
        
    X1_3 = conv1(X1_2)
    X2_3 = conv1(X2_2)
    
    X1_4 = conv2(X1_3)
    X2_4 = conv2(X2_3)
    
    X1_4 = concatenate([X1_4, X4], axis=-1)

    X2_4 = concatenate([X2_4, X4], axis=-1)

    
    X1_5 = conv3(X1_4)
    X2_5 = conv3(X2_4)
    
    X6 = concatenate([X1_5, X2_5], axis=-1)
    
    X = Dense(1,activation='sigmoid')(X6)
    model = Model(inputs = input_shape, outputs = X, name='HappyModel')

        
        
    return model
    
     
    
    
    
    
    
    
    
    