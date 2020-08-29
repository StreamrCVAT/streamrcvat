import pandas as pd 
import numpy as np 
import cv2
import os
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt
import math 

from global_params import *
from stripping_edges import *
from create_models import *

from numpy.random import seed
seed(1)


def min_max_scalar(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

def batch_train(side_name):
    BASE_SIZE = 32 #number of images to train for base model

    #real frame size
    frame_width = FRAME_WW
    frame_height = FRAME_HH

    #paths to coordinate dirs
    yolo_output_dir = 'D:\\smart_space_lab\\Intel_Annotation\\code\\modelA\\yolo_output'
    human_annotated_dir = 'D:\\smart_space_lab\\Intel_Annotation\\frames\\annotated frames coor (1 to 499)'
    #path to frames dir
    frames_dir = 'D:\\smart_space_lab\\Intel_Annotation\\frames\\video_trim (frames)'
    print('---------------------PATH to DIRS SET---------------------------')

    #create the model
    side_model = construct_modelB_cnn(MODEL_HEIGHT, MODEL_WIDTH, 3, True)
    print('----------------------4 MODELS CONSTRUCTED-----------------------')

    print('SUMMARY OF RIGTH MODEL')
    print(side_model.summary())

    #compiling all models 
    side_model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])

    X_train_image = []
    X_train_yolo = []
    y_train = []
    inc = 0
    
    for filename in os.listdir(yolo_output_dir):
        if(filename=='readme.txt'):
            continue
        print('processing', filename)
        path1 = os.path.join(yolo_output_dir, filename) #yolo outputs
        path2 = os.path.join(human_annotated_dir, filename) #human outputs
        path3 = os.path.join(frames_dir, filename[:-3]+'jpg') #image frames

        file1 = open(path1, 'r')
        file2 = open(path2, 'r')
        line1 = file1.readline()
        line2 = file2.readline()
        line1 = list(map(float, line1.split()[1:])) #[ymin xmin ymax xmax]
        line2 = list(map(float, line2.split()[1:])) #[ymin xmin ymax xmax]

        #read the original frame
        image = cv2.imread(path3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # scale pixel values to [0, 1]
        image = image.astype('float32')
        image /= 255.0
        strips = strip(image, line1[1], line1[0], line1[3], line1[2]) #args: img, xmin, ymin, xmax, ymax
        if(side_name == 'right'):
            right_part = cv2.resize(strips[0], (MODEL_HEIGHT, MODEL_WIDTH))
            X_train_image.append(right_part)
            X_train_yolo.append(line1[3])
            y_train.append(line2[3]-line1[3]) #here o/p is the error(human - yolo)
        elif(side_name == 'bottom'):
            bottom_part = cv2.resize(strips[1], (MODEL_HEIGHT, MODEL_WIDTH))
            X_train_image.append(bottom_part)
            X_train_yolo.append(line1[2])
            y_train.append(line2[2]-line1[2]) #here o/p is the error(human - yolo)
        elif(side_name == 'left'):
            left_part = cv2.resize(strips[2], (MODEL_HEIGHT, MODEL_WIDTH))
            X_train_image.append(left_part)
            X_train_yolo.append(line1[1])
            y_train.append(line2[1]-line1[1]) #here o/p is the error(human - yolo)
        else: #top side
            top_part = cv2.resize(strips[3], (MODEL_HEIGHT, MODEL_WIDTH))
            X_train_image.append(top_part)
            X_train_yolo.append(line1[0])
            y_train.append(line2[0]-line1[0]) #here o/p is the error(human - yolo)

        inc += 1
        if(inc == BASE_SIZE):
            break

    X_train_image = np.array(X_train_image)
    X_train_yolo = np.array(X_train_yolo)
    y_train = np.array(y_train)
    print(X_train_image.shape)
    print(X_train_yolo.shape)
    print(y_train.shape)

    hist = side_model.fit(X_train_image[:32,:,:,:], y_train[:32], epochs=15)

    return side_model
