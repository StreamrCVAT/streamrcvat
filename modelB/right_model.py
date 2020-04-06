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
from keras.layers import BatchNormalization
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

from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

def construct_modelB_cnn(height, width, depth, regres=False): #height, width, depth are the input dimensions of the model

    # 1 Convolutional layer
    inp = Input(shape=(width,height,depth))
    conv1 = Conv2D(32, kernel_size=3, padding='same')(inp)
    # batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(3,3))(act1)
    

    # 2 Convolutional layer
    # conv2 = Conv2D(128, kernel_size=3, padding='same')(pool1)
    # act2 = Activation('relu')(conv2)
    # pool2 = MaxPooling2D(pool_size=(3,3))(act2)

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
    if(regres == True):
        output_regres = Dense(1, activation='linear')(output)
        output = output_regres
        return Model(inputs=[inp], outputs=[output])
    return inp, output

def construct_modelB_line(regres=False):
    #single neuron
    inp = Input(shape=(1,))
    hidden1 = Dense(1)(inp)

    if(regres == True):
        act4 = Activation('linear')(hidden1)
        model = Model(inputs=inp, outputs=act4)
        return model
    act1 = Activation('relu')(hidden1)
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
    # print(w_r, h_r)
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

def visualize_strips(image, xmin, ymin, xmax, ymax):
    print('image shape', (image).shape) # (Y, X, DEPTH=3)

    start_point = (xmin, ymin) 
    end_point = (xmax, ymax) 
    color = (255, 0, 0) 
    thickness = 2
    annotated = cv2.rectangle(image, start_point, end_point, color, thickness)

    cv2.imshow('YOLO Image',annotated)   
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

def batch_train():
    BASE_SIZE = 16+450 #number of images to train for base model

    #real frame size
    frame_width = 1280
    frame_height = 720

    #paths to coordinate dirs
    yolo_output_dir = 'D:\\smart_space_lab\\Intel_Annotation\\code\\modelA\\yolo_output'
    human_annotated_dir = 'D:\\smart_space_lab\\Intel_Annotation\\frames\\annotated frames coor (1 to 499)'
    #path to frames dir
    frames_dir = 'D:\\smart_space_lab\\Intel_Annotation\\frames\\video_trim (frames)'
    print('---------------------PATH to DIRS SET---------------------------')

    #create the model
    right_model = construct_modelB(frame_height//6, frame_width//10, 3)
    print('----------------------4 MODELS CONSTRUCTED-----------------------')

    print('SUMMARY OF RIGTH MODEL')
    print(right_model.summary())

    #compiling all models 
    right_model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])

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

        right_part = cv2.resize(strips[0], (frame_height//6, frame_width//10))

        X_train_image.append(right_part)
        # X_train_yolo.append(line1[3]/frame_width)
        X_train_yolo.append(line1[3])
        y_train.append(line2[3])


        inc += 1
        if(inc == BASE_SIZE):
            break

    X_train_image = np.array(X_train_image)
    X_train_yolo = np.array(X_train_yolo)
    y_train = np.array(y_train)
    print(X_train_image.shape)
    print(X_train_yolo.shape)
    print(y_train.shape)

    hist = right_model.fit([X_train_image[:32,:,:,:], X_train_yolo[:32]], y_train[:32], epochs=15)
    # hist = right_model.fit(X_train_yolo[:32], y_train[:32], epochs=60)
    # scores = right_model.evaluate([X_train_image, X_train_yolo], y_train)
    # print(scores)
    y_pred = right_model.predict([X_train_image[:32,:,:,:], X_train_yolo[:32]])
    # y_pred = right_model.predict(X_train_yolo[:16])
    y_pred =y_pred.T
    error = abs(y_train[:32]-y_pred[0])
    # print(y_train.shape)
    # print(y_pred.shape)
    print(error.astype('int'))

    # hist = right_model.fit([X_train_image[16:,:,:,:], X_train_yolo[16:]], y_train[16:], epochs=15)
    # scores = right_model.evaluate([X_train_image, X_train_yolo], y_train)
    # print(scores)
    y_pred = right_model.predict([X_train_image[32:,:,:,:], X_train_yolo[32:]])
    # y_pred = right_model.predict(X_train_yolo[16:])
    y_pred =y_pred.T
    error = abs(y_train[32:]-y_pred[0])
    # print(y_train.shape)
    # print(y_pred.shape)
    print(error.astype('int'))

    # Plot history: MSE
    plt.plot(hist.history['loss'], label='Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch no')
    plt.legend(loc="upper left")
    plt.show()



def pipeline():
    #paths to coordinate dirs
    yolo_output_dir = 'D:\\smart_space_lab\\Intel_Annotation\\code\\modelA\\yolo_output'
    human_annotated_dir = 'D:\\smart_space_lab\\Intel_Annotation\\frames\\annotated frames coor (1 to 499)'
    #path to frames dir
    frames_dir = 'D:\\smart_space_lab\\Intel_Annotation\\frames\\video_trim (frames)'
    print('---------------------PATH to DIRS SET---------------------------')

    #real frame size
    frame_width = 1280
    frame_height = 720

    #create the model
    right_model = construct_modelB(frame_height//4, frame_width//4, 3)
    print('----------------------4 MODELS CONSTRUCTED-----------------------')

    print('SUMMARY OF RIGTH MODEL')
    print(right_model.summary())

    #compiling all models 
    right_model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])
    print('--------------------4 MODELS COMPILED-----------------------------')

    correct_model_a = 0
    correct_model_b_right = 0

    history = {'loss':[], 'mae':[]}
    for filename in os.listdir(yolo_output_dir):
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
            image /= 255.0
            strips = strip(image, line1[1], line1[0], line1[3], line1[2]) #args: img, xmin, ymin, xmax, ymax
                    
            if(line1[3] != line2[3]): #xmax do not match (right line eqn)
                right_part = cv2.resize(strips[0], (frame_height//4, frame_width//4))
                # right_pre = right_model.predict([[right_part], [min_max_scalar(line1[3], 0, right_part.shape[1])]])
                right_pre = right_model.predict([[right_part], [line1[3]]])
                # right_pre = right_model.predict([[right_part]])
                if(right_pre != line2[3]): #modelB wrongly predicts
                    print(filename, 'Model B predicts RIGHT WRONG - FITTING Model B')
                    hist = right_model.fit([[right_part], [line1[3]]], [line2[3]])
                    history['loss'].append(hist.history['loss'][0])
                    history['mae'].append(hist.history['mae'][0])
                else:
                    print(filename, 'Model B predicts RIGHT CORRECT')
                    correct_model_b_right += 1

    print('correct_model_a', correct_model_a)
    print('correct_model_b_right', correct_model_b_right)
    print(history)
    # Plot history: MAE
    plt.plot(history['loss'], label='loss')
    plt.ylabel('loss value')
    plt.xlabel('Sample #')
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    # modelb = construct_modelB(32, 32, 3)
    # print(modelb.summary())   
    # plot_model(modelb, to_file='modelb.png')
    # image = cv2.imread('D:\\smart_space_lab\\Intel_Annotation\\code\\images\\video_trim 041.jpg')
    # visualize_strips(image, 624, 155, 960, 320) #155, 624, 320, 960
    # pipeline()
    batch_train()
    # model = construct_modelB(180, 320, 3)
    # model.summary()
    
