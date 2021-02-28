# This the implementation of ModelB
# This model learns the errors associated with the annotated frams

import pandas as pd 
import numpy as np 
import cv2
import os
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.utils import plot_model
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

from numpy.random import seed
seed(1)

from global_params import *
from batch_training import *

right_model
bottom_model
left_model 
top_model 

def retrain_model(side_name, image_path, yolo_coor, human_coor):

    global right_model
    global bottom_model
    global left_model 
    global top_model 

    #read the original frame
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # scale pixel values to [0, 1]
    image = image.astype('float32')
    strips = strip(image, yolo_coor[1], yolo_coor[0], yolo_coor[3], yolo_coor[2]) #args: img, xmin, ymin, xmax, ymax
    strips[0] /= 255.0
    strips[1] /= 255.0
    strips[2] /= 255.0
    strips[3] /= 255.0

    # checkpoint
    es = EarlyStopping(monitor='mae', mode='min', patience=10, restore_best_weights=True, verbose=1)

    if(side_name == 'bottom'):
        bottom_part = cv2.resize(strips[1], (MODEL_HEIGHT, MODEL_WIDTH))
        bottom_pre = bottom_model.predict([[bottom_part]])

        bottom_model.fit([[bottom_part]], [[human_coor[2]-yolo_coor[2]]], epochs=10, callbacks=[es])

    elif(side_name == 'left'):
        left_part = cv2.resize(strips[2], (MODEL_HEIGHT, MODEL_WIDTH))
        left_pre = left_model.predict([[left_part]])

        left_model.fit([[left_part]], [[human_coor[1]-yolo_coor[1]]], epochs=10, callbacks=[es])

    elif(side_name == 'top'):
        top_part = cv2.resize(strips[3], (MODEL_HEIGHT, MODEL_WIDTH))
        top_pre = top_model.predict([[top_part]])
                
        top_model.fit([[top_part]], [[human_coor[0]-yolo_coor[0]]], epochs=10, callbacks=[es])

    else: #right side
        right_part = cv2.resize(strips[0], (MODEL_HEIGHT, MODEL_WIDTH))
        right_pre = right_model.predict([[right_part]])

        right_model.fit([[right_part]], [[human_coor[3]-yolo_coor[3]]], epochs=10, callbacks=[es])
                

def check_error(image_path, yolo_coor, modelB_coor, human_coor): #[ymin xmin ymax xmax]
    
    #check for bottom side
    if((modelB_coor[2] <= yolo_coor[2]-3) or (modelB_coor[2] >= yolo_coor[2]+3)): #modelB wrongly predicts
        retrain_model('bottom', image_path, yolo_coor, human_coor)
    
    #check for left side
    if((modelB_coor[1] <= yolo_coor[1]-3) or (modelB_coor[1] >= yolo_coor[1]+3)): #modelB wrongly predicts
        retrain_model('left', image_path, yolo_coor, human_coor)

    #check for top side
    if((modelB_coor[0] <= yolo_coor[0]-3) or (modelB_coor[0] >= yolo_coor[0]+3)): #modelB wrongly predicts
        retrain_model('top', image_path, yolo_coor, human_coor)

    #check for right side
    if((modelB_coor[3] <= yolo_coor[3]-3) or (modelB_coor[3] >= yolo_coor[3]+3)): #modelB wrongly predicts
        retrain_model('right', image_path, yolo_coor, human_coor)


def modelB_prediction(image_path, yolo_coor):

    predictions = [0, 0, 0, 0]

    #read the original frame
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # scale pixel values to [0, 1]
    image = image.astype('float32')
    # image /= 255.0
    strips = strip(image, yolo_coor[1], yolo_coor[0], yolo_coor[3], yolo_coor[2]) #args: img, xmin, ymin, xmax, ymax
    strips[0] /= 255.0
    strips[1] /= 255.0
    strips[2] /= 255.0
    strips[3] /= 255.0

    #right side
    right_part = cv2.resize(strips[0], (MODEL_HEIGHT, MODEL_WIDTH))
    right_pre = right_model.predict([[right_part]])
    predictions[3] = yolo_coor[3]+right_pre[0][0]

    #bottom side
    bottom_part = cv2.resize(strips[1], (MODEL_HEIGHT, MODEL_WIDTH))
    bottom_pre = bottom_model.predict([[bottom_part]])
    predictions[2] = yolo_coor[2]+bottom_pre[0][0]

    #left side
    left_part = cv2.resize(strips[2], (MODEL_HEIGHT, MODEL_WIDTH))
    left_pre = left_model.predict([[left_part]])
    predictions[1] = yolo_coor[1]+left_pre[0][0]

    #top side
    top_part = cv2.resize(strips[3], (MODEL_HEIGHT, MODEL_WIDTH))
    top_pre = top_model.predict([[top_part]])
    predictions[0] = yolo_coor[0]+top_pre[0][0]
                
    return predictions 


def main():

    global right_model
    global bottom_model
    global left_model 
    global top_model 

    right_model = batch_train('right')
    bottom_model = batch_train('bottom')
    left_model = batch_train('left')
    top_model = batch_train('top')

    finalCoordinatePath = os.path.dirname(os.path.realpath(__file__)) + "\\data\\" + FINAL_UI_OUTPUT_PATH
    frame_number = BATCH_SIZE

    while(True):
        if (len(os.listdir(finalCoordinatePath)) == frame_number):
            check_error(image_path, yolo_coor, modelB_coor, human_coor): #[ymin xmin ymax xmax] 
        


    return True
    
if __name__ == '__main__':
    main()
