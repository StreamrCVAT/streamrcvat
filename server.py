# This the implementation of ModelB
# This model learns the errors associated with the annotated frams

import pandas as pd 
import numpy as np 
import cv2
import os
 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt
import math 

from numpy.random import seed
seed(1)

from models import batch_training
from models import create_models
from models import global_params
from models import stripping_edges
from utils import yoloTracker

from main import *

from flask import Flask, request, render_template, jsonify

# app = Flask()
app = Flask(__name__, template_folder="templates")

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
frame_path = ABSOLUTE_PATH+"\\data\\"+FRAMES_PATH+'\\'
yolo_coor_path = ABSOLUTE_PATH+"\\data\\"+YOLO_OUTPUT_TRACKED_PATH+'\\'
modelB_coor_path = ABSOLUTE_PATH+"\\data\\"+MODEL_B_OUTPUT_PATH+'\\'
human_coor_path = ABSOLUTE_PATH+"\\data\\"+FINAL_UI_OUTPUT_PATH+'\\'

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
                

def fix_errors(image_path, yolo_coor, modelB_coor, human_coor): #[ymin xmin ymax xmax]
    
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


def modelB_prediction(image_path, yolo_coor, next_modelB_path):

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
    
    # write the modelB predictions to modelB prediction folder
    with open(next_modelB_path, 'w') as new_file:
        new_file.write('car '+' '.join(map(str, predictions))+'\n')


def createBaseModels():
    
    global right_model
    global bottom_model
    global left_model 
    global top_model 
    
    right_model = batch_training.batch_train('right')
    bottom_model = batch_training.batch_train('bottom')
    left_model = batch_training.batch_train('left')
    top_model = batch_training.batch_train('top')

@app.route("/")
def home():
    return render_template('form.html')

@app.route("/trigger", methods=['POST'])
def triggerAPI():
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            frame_number = request_data['frame_number']
            frame_filename = request_data['frame_filename']
            print(frame_number, frame_filename)
            # frame_number = int(request.form['frame_number'])
            # frame_filename = request.form['frame_filename']
            
            if(frame_number == 1):
                yolo_next_frame = yoloTracker.trackNextObject(0, frame_filename)
            yolo_next_frame = yoloTracker.trackNextObject(frame_number, frame_filename)

            if (frame_number < BATCH_SIZE):
                helper.copy_file_to(yolo_next_frame, modelB_coor_path + yolo_next_frame.split("\\")[-1])
            
            if(frame_number >= BATCH_SIZE):
                # Image path 
                # yolo coor
                # modelB coor
                # human coor

                cur_image_path = frame_path+'frame-' + str(frame_number).zfill(3) + '.jpg'
                next_image_path = frame_path+'frame-' + str(frame_number+1).zfill(3) + '.jpg'
                cur_yolo_path = yolo_coor_path+frame_filename
                next_yolo_path = getFilesPathAsList(yolo_coor_path)[int(frame_number)]
                cur_modelB_path = modelB_coor_path+frame_filename
                next_modelB_path = modelB_coor_path+'\\'+next_yolo_path.split('\\')[-1]
                cur_human_path = human_coor_path+frame_filename

                cur_yolo_coor = list(map(float, open(cur_yolo_path, 'r').readline().split()[1:])) #[ymin xmin ymax xmax]
                next_yolo_coor = list(map(float, open(next_yolo_path, 'r').readline().split()[1:]))
                cur_modelB_coor = list(map(float, open(cur_modelB_path, 'r').readline().split()[1:]))
                cur_human_coor = list(map(float, open(cur_human_path, 'r').readline().split()[1:]))
                
                if (frame_number == BATCH_SIZE):
                    # createYOLOTracker(frame_filename)
                    createBaseModels()
                    return {'message': "32=Success"}
                else:
                    fix_errors(cur_image_path, cur_yolo_coor, cur_modelB_coor, cur_human_coor)
                modelB_prediction(next_image_path, next_yolo_coor, next_modelB_path)

                return {'message': ">=32-Success"}

            else:
                return {'message': "<32-Success"}

        except:
            print("Excep Error")
            return {'message': "Error"}
    return {'message': "GET-Success"}
    

if (__name__=='__main__'):
    app.run(debug=True)
