# This the implementation of ModelB
# This model learns the errors associated with the annotated frams

import pandas as pd 
import numpy as np 
import cv2
import os
#import tensorflow as tf
#from keras.utils import plot_model
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Activation
#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt
import math 

def construct_modelB_cnn(height, width, depth): #height, width, depth are the input dimensions of the model

    # 1 Convolutional layer
    inp = Input(shape=(width,height,depth))
    conv1 = Conv2D(64, kernel_size=3, padding='same')(inp)
    act1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(act1)

    # Flatten layer
    flat2 = Flatten()(pool1)

    #Dense Layers
    hidden3 = Dense(16)(flat2)
    act3 = Activation('relu')(hidden3)
    output = act3
    #Output Layer 
    # hidden4 = Dense(1)(act3)
    # act4 = Activation('linear')(hidden4)

    # model = Model(inputs=input, outputs=act4)
    # return model
    return inp, output

def construct_modelB_line():
    #single neuron
    inp = Input(shape=(1,))
    hidden1 = Dense(1)(inp)
    act1 = Activation('linear')(hidden1)

    # model = Model(inputs=input, outputs=act1)
    # return model
    output = act1
    return inp, output

def construct_modelB(height, width, depth):
    cnn_model_input, cnn_model_output = construct_modelB_cnn(height, width, depth)
    line_model_input, line_model_output = construct_modelB_line()
    
    merge = concatenate([cnn_model_output, line_model_output])
    
    hidden1 = Dense(8)(merge)
    act1 = Activation('relu')(hidden1)
    hidden2 = Dense(1)(act1)
    act2 = Activation('linear') (hidden2)
    model = Model(inputs=[cnn_model_input, line_model_input], outputs=[act2])
    return model

def strip(image, xmin, ymin, xmax, ymax):
    #return strip of the 4 edges
    #image is 3D numpy array, coordinates are output of YoLo model
    width = abs(xmax-xmin)
    height = abs(ymax-ymin)
    w_r = width//2 # selecting 1/2th segment from either side of the predicted line
    h_r = height//2 # selecting 1/2th segment from either side of the predicted line
    print(w_r, h_r)
    ymin_t = int(max(0, ymin-h_r))
    ymin_b = int(max(0, ymin+h_r))
    ymax_t = int(max(0, ymax-h_r)) 
    ymax_b = int(max(0, ymax+h_r))
    xmin_l = int(max(0, xmin-w_r)) 
    xmin_r = int(max(0, xmin+w_r))
    xmax_l = int(max(0, xmax-w_r)) 
    xmax_r = int(max(0, xmax+w_r))
    #print(image.shape)
    right_strip = image[ymin_t:ymax_b, xmax_l:xmax_r, :]
    bottom_strip = image[ymax_t:ymax_b, xmin_l:xmax_r, :]
    left_strip = image[ymin_t:ymax_b, xmin_l:xmin_r, :]
    top_strip = image[ymin_t:ymin_b, xmin_l:xmax_r, :]

    return [right_strip, bottom_strip, left_strip, top_strip]

def visualize_strips(image):
    print('image shape', (image).shape) # (Y, X, DEPTH=3)
    cv2.imshow('Original Image',image)   
    cv2.waitKey(0)
    # car 155 624 320 960
    ret = strip(image, 624, 155, 960, 320)
    rs = np.asarray(ret[0]) # Right Strip
    bs = np.asarray(ret[1]) # Bottom Strip
    ls = np.asarray(ret[2]) # Left Strip
    ts = np.asarray(ret[3]) # Top Strip
    cv2.imshow('Right Strip',rs)
    cv2.waitKey(0)
    cv2.imshow('Bottom Strip',bs)
    cv2.waitKey(0)
    cv2.imshow('Left Strip',ls)
    cv2.waitKey(0)
    cv2.imshow('Top Strip',ts)
    cv2.waitKey(0)

def min_max_scalar(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

def pipeline():
    #paths to coordinate dirs
    yolo_output_dir = 'E:\\General\\Personal\\Pro\\Intel Annotation pro\\Repo\\Annotation\\modelB\\yolo_output'
    human_annotated_dir = 'E:\\General\\Personal\\Pro\\Intel Annotation pro\\Repo\\Annotation\\modelB\\annotated frames coor (1 to 499)'
    #path to frames dir
    frames_dir = 'E:\\General\\Personal\\Pro\\Intel Annotation pro\\Repo\\Annotation\\video_trim (frames)'
    #real frame size
    frame_width = 1280
    frame_height = 720

    #create the model
    right_model = construct_modelB(frame_height//4, frame_width//4, 3)
    left_model = construct_modelB(frame_height//4, frame_width//4, 3)
    bottom_model = construct_modelB(frame_height//4, frame_width//4, 3)
    top_model = construct_modelB(frame_height//4, frame_width//4, 3)
    print(top_model.summary())

    #compiling all models 
    right_model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])
    bottom_model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])
    left_model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])
    top_model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])

    for filename in os.listdir(yolo_output_dir):
        if(filename=='readme.txt'):
            continue
        path1 = os.path.join(yolo_output_dir, filename) #yolo outputs
        path2 = os.path.join(human_annotated_dir, filename) #human outputs
        path3 = os.path.join(frames_dir, filename[:-3]+'jpg') #image frames

        file1 = open(path1, 'r')
        file2 = open(path2, 'r')
        line1 = file1.readline()
        line2 = file2.readline()
        line1 = list(map(float, line1.split()[1:])) #[ymin xmin ymax xmax]
        line2 = list(map(float, line2.split()[1:])) #[ymin xmin ymax xmax]

        if(line1 == line2): #no error
            print('Yolo prediction correct')
            continue
        else:
            #read the original frame
            image = cv2.imread(path3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # scale pixel values to [0, 1]
            image = image.astype('float32')
            image /= 255.0
            strips = strip(image, line1[1], line1[0], line1[3], line1[2]) #args: img, xmin, ymin, xmax, ymax

            if(line1[0] != line2[0]): #ymin do not match (top line eqn)
                top_part = cv2.resize(strips[3], (frame_height//4, frame_width//4))
                #print(top_part.shape)
                top_pre = top_model.predict([[top_part], [min_max_scalar(line1[0], 0, int(top_part.shape[0]))]]) #scale with height of crop part
                #top_pre = top_model.predict([np.array(top_part)])
                if(top_pre != line2[0]): #modelB wrongly predicts
                    top_model.fit([[top_part], [min_max_scalar(line1[0], 0, top_part.shape[0])]], [line2[0]])

            if(line1[1] != line2[1]): #xmin do not match (left line eqn)
                left_part = cv2.resize(strips[2], (frame_height//4, frame_width//4))
                left_pre = left_model.predict([[left_part], [min_max_scalar(line1[1], 0, left_part.shape[1])]]) #scale with width of crop part
                if(left_pre != line2[1]): #modelB wrongly predicts
                    left_model.fit([[left_part], [min_max_scalar(line1[1], 0, left_part.shape[1])]], [line2[1]])

            if(line1[2] != line2[2]): #ymax do not match (bot line eqn)
                bot_part = cv2.resize(strips[1], (frame_height//4, frame_width//4))
                bot_pre = bottom_model.predict([[bot_part], [min_max_scalar(line1[2], 0, bot_part.shape[0])]])
                if(bot_pre != line2[2]): #modelB wrongly predicts
                    bottom_model.fit([[bot_part], [min_max_scalar(line1[2], 0, bot_part.shape[0])]], [line2[2]])
                    
            if(line1[3] != line2[3]): #xmax do not match (right line eqn)
                right_part = cv2.resize(strips[0], (frame_height//4, frame_width//4))
                right_pre = right_model.predict([[right_part], [min_max_scalar(line1[3], 0, right_part.shape[1])]])
                if(right_pre != line2[3]): #modelB wrongly predicts
                    right_model.fit([[right_part], [min_max_scalar(line1[3], 0, right_part.shape[1])]], [line2[3]])

        print(filename, line1, line2)

if __name__ == "__main__":
    # modelb = construct_modelB(32, 32, 3)
    # print(modelb.summary())   
    # plot_model(modelb, to_file='modelb.png')
    # image = cv2.imread('D:\\smart_space_lab\\Intel_Annotation\\code\\images\\video_trim 041.jpg')
    # visualize_strips(image)
    pipeline()
    # model = construct_modelB(180, 320, 3)
    # model.summary()
    
