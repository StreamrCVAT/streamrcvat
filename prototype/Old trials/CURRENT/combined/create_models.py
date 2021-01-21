
import pandas as pd 
import numpy as np
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
import math 

from numpy.random import seed
seed(1)


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
    hidden3 = Dense(64)(flat2)
    act3 = Activation('relu')(hidden3)
    hidden4 = Dense(16)(act3)
    act4 = Activation('relu')(hidden4)
    output = act4
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

def construct_dnn(height, width, depth):
    inp = Input(shape=(width, height, depth))
    flat = Flatten()(inp)
    hid1 = Dense(128, activation='relu')(flat)
    hid2 = Dense(64, activation='relu')(hid1)
    hid3 = Dense(32, activation='relu')(hid2)
    hid4 = Dense(16, activation='relu')(hid3)
    op = Dense(1, activation='linear')(hid4)

    model = Model(inputs=inp, outputs=op)
    return model
