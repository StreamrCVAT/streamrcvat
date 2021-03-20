
import pandas as pd 
import numpy as np
import os
 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
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
import math 

from numpy.random import seed
seed(1)


def construct_modelB_cnn(height, width, depth): #height, width, depth are the input dimensions of the model

    # 1 Convolutional layer
    inp = Input(shape=(width,height,depth))
    conv1 = Conv2D(32, kernel_size=3, padding='same')(inp)
    # batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(3,3))(act1)

    # Flatten layer
    flat2 = Flatten()(pool1)

    #Dense Layers
    hidden3 = Dense(64)(flat2)
    act3 = Activation('relu')(hidden3)
    hidden4 = Dense(16)(act3)
    act4 = Activation('relu')(hidden4)
    output = act4

    output_regres = Dense(1, activation='linear')(output)
    output = output_regres
    return Model(inputs=[inp], outputs=[output])
