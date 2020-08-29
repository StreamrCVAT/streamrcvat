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
from performance_visualize import *
from right_model import *
from bottom_model import *
from left_model import *
from top_model import *


if __name__ == "__main__":
    right_pipeline()
    bottom_pipeline()
    left_pipeline()
    top_pipeline()

    