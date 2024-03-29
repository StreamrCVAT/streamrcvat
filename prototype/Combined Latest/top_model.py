# This the implementation of ModelB
# This model learns the errors associated with the annotated frams

import pandas as pd 
import numpy as np 
import cv2
import os
import csv
from keras.utils import plot_model
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
#from keras.utils import plot_model
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
from performance_visualize import *


def top_pipeline(YOLO_OUTPUT_DIR, HUMAN_ANNOTATED_DIR, FRAMES_DIR):
    performance = [['human_line', 'YOLO line', 'ModelB line', 'YOLO error', 'ModelB error']]
    #paths to coordinate dirs
    yolo_output_dir = YOLO_OUTPUT_DIR
    human_annotated_dir = HUMAN_ANNOTATED_DIR
    #path to frames dir
    frames_dir = FRAMES_DIR
    print('---------------------PATH to DIRS SET---------------------------')

    #real frame size
    frame_width = FRAME_WW
    frame_height = FRAME_HH

    top_model = batch_train("top", YOLO_OUTPUT_DIR, HUMAN_ANNOTATED_DIR, FRAMES_DIR)

    # checkpoint
    es = EarlyStopping(monitor='mae', mode='min', patience=10, restore_best_weights=True, verbose=1)

    correct_model_a = 0
    correct_model_b_top = 0
    predictions = list()
    yolo_human_errors = []
    history = {'loss':[], 'mae':[]}
    for filename in os.listdir(yolo_output_dir):
        perf = [0, 0, 0, 0, 0]
        if(filename=='readme.txt'):
            continue

        print('-----Processing', filename, '-----')

        path1 = os.path.join(yolo_output_dir, filename) #yolo outputs
        path2 = os.path.join(human_annotated_dir, filename) #human outputs
        path3 = os.path.join(frames_dir, filename[:-3]+'jpg') #image frames

        file1 = open(path1, 'r')
        file2 = open(path2, 'r')
        line1 = file1.readline()
        line2 = file2.readline()
        line1 = list(map(float, line1.split()[1:])) #[ymin xmin ymax xmax]
        line2 = list(map(float, line2.split()[1:])) #[ymin xmin ymax xmax]

        print(filename, line1, line2)

        if(line1 == line2): #no error
            print(filename, '- Yolo prediction CORRECT')
            correct_model_a += 1
            continue
        else:
            print(filename, '- Yolo prediction WRONG')
            #read the original frame
            image = cv2.imread(path3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # scale pixel values to [0, 1]
            image = image.astype('float32')
            # image /= 255.0
            strips = strip(image, line1[1], line1[0], line1[3], line1[2]) #args: img, xmin, ymin, xmax, ymax
            strips[0] /= 255.0
            strips[1] /= 255.0
            strips[2] /= 255.0
            strips[3] /= 255.0
            if(line1[0] != line2[0]): #ymin do not match (top line eqn)
                yolo_human_errors.append(line1[0]-line2[0])
                top_part = cv2.resize(strips[3], (MODEL_HEIGHT, MODEL_WIDTH))
                top_pre = top_model.predict([[top_part]])
                predictions.append(line2[0]-(line1[0]+top_pre[0][0]))

                perf[0] = line2[0] #human line
                perf[1] = line1[0] #yolo line
                perf[2] = line1[0]+top_pre[0][0] #ModelB output line
                perf[3] = abs(line1[0]-line2[0]) #YOLO error
                perf[4] = abs(perf[2]-line2[0]) #modelB error
                performance.append(perf)

                if(((top_pre[0][0]+line1[0]) <= (line2[0]-3)) or ((top_pre[0][0]+line1[0]) >= (line2[0]+3))): #modelB wrongly predicts
                    print(filename, 'Model B predicts top WRONG - FITTING Model B')
                    hist = top_model.fit([[top_part]], [[line2[0]-line1[0]]], epochs=10, callbacks=[es])
                    history['loss'].append(hist.history['loss'][0])
                    history['mae'].append(hist.history['mae'][0])
                else:
                    print(filename, 'Model B predicts top CORRECT')
                    history['loss'].append(-1)
                    history['mae'].append(-1)
                    correct_model_b_top += 1

    print('correct_model_a', correct_model_a)
    print('correct_model_b_top', correct_model_b_top)
    print(predictions)
    plt.hist(predictions, bins=10)
    plt.ylabel('frame counts')
    plt.xlabel('error')
    plt.title('ModelB errors vs Number of frames')
    plt.show()
    
    plt.plot(predictions)
    plt.ylabel('error')
    plt.xlabel('frame #')
    plt.title('ModelB error generated for the sequence of frames')
    
    plt.show()
    # print(history)
    # Plot history: MAE
    plt.plot(history['loss'], label='loss')
    plt.ylabel('loss value')
    plt.xlabel('Sample #')
    plt.legend(loc="upper top")
    plt.title('Model Training error at every frame')
    plt.show()

    print(yolo_human_errors)
    save_i = input("Save the model? ")
    if(save_i == 'y' or save_i == 'Y'):
        # serialize model to JSON
        model_json = top_model.to_json()
        with open("top_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        top_model.save_weights("top_model.h5")
        print("Saved model to disk")
    
    with open("top_performance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(performance)

    visualize_performance(performance) # only for errors 
