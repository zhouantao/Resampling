# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:21:14 2018

@author: 84583
"""

from glob import glob
import numpy as np


def load_data():

     
     train_0,val_0,test_0 = returndata('/home/dell/zhou/dagai/data/down50/55/',0)
     train_1,val_1,test_1= returndata('/home/dell/zhou/dagai/data/jpeg/55/',1)

     
     train = np.concatenate([train_0,train_1],axis = 0)
     val = np.concatenate([val_0,val_1],axis = 0)
     test = np.concatenate([test_0,test_1],axis = 0)
    
           
     return train,val,test
     
def returndata(path,label):
    path_image = []
    for i in range(len(glob(path + '*'))):
        image_name = path + str(i+1) + '.jpg'
        path_image.append(image_name)
    path_image = np.array(path_image)
    if label == 0:
        data_label =  np.zeros(len(path_image),dtype=int)
    else:
        data_label = np.ones(len(path_image),dtype=int) * label
    data = np.hstack((path_image.reshape(-1,1),data_label.reshape(-1,1))) 
    return data[:25000],data[25000:30000],data[30000:]
    

    
