# This the implementation of ModelB
# This model learns the errors associated with the annotated frams

import pandas as pd 
import numpy as np 
from keras.utils import plot_model
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Activation
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from PIL import Image
    
def strip(image, xmin, ymin, xmax, ymax):
    #return strip of the 4 edges
    #image is 3D numpy array, coordinates are output of YoLo model
    width = abs(xmax-xmin)
    height = abs(ymax-ymin)
    w_r = width//4 # selecting 1/4th segment from either side of the predicted line
    h_r = height//4 # selecting 1/4th segment from either side of the predicted line
    print(w_r, h_r)
    ymin_t = max(0, ymin-h_r)
    ymin_b = max(0, ymin+h_r)
    ymax_t = max(0, ymax-h_r) 
    ymax_b = max(0, ymax+h_r)
    xmin_l = max(0, xmin-w_r) 
    xmin_r = max(0, xmin+w_r)
    xmax_l = max(0, xmax-w_r) 
    xmax_r = max(0, xmax+w_r) 

    right_strip = image[ymin_t:ymax_b, xmax_l:xmax_r, :]
    bottom_strip = image[ymax_t:ymax_b, xmin_l:xmax_r, :]
    left_strip = image[ymin_t:ymax_b, xmin_l:xmin_r, :]
    top_strip = image[ymin_t:ymin_b, xmin_l:xmax_r, :]

    return [right_strip, bottom_strip, left_strip, top_strip]


if __name__ == "__main__":
    image = load_img('D:\\smart_space_lab\\Intel_Annotation\\code\\images\\video_trim 041.jpg')
    # pyplot.imshow(image)
    # pyplot.show()
    image = img_to_array(image)
    print(image.shape) # (Y, X, DEPTH=3)
    print(image)    
    print('----------')
    # car 155 624 320 960
    ret = strip(image, 624, 155, 960, 320)
    img = Image.fromarray(ret[1], 'RGB')
    print(ret)
    pyplot.imshow(img)
    pyplot.show()
