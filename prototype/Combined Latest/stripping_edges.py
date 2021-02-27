import pandas as pd 
import numpy as np 
import cv2
import os

from matplotlib import pyplot
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt
import math 

from numpy.random import seed
seed(1)

def strip(image, xmin, ymin, xmax, ymax):
    #return strip of the 4 edges
    #image is 3D numpy array, coordinates are output of YoLo model
    frame_w = image.shape[1]
    frame_h = image.shape[0]

    width = abs(xmax-xmin)
    height = abs(ymax-ymin)
    w_r = width//2 # selecting 1/2th segment from either side of the predicted line
    h_r = height//2 # selecting 1/2th segment from either side of the predicted line

    ymin_t = int(max(0, ymin-h_r))
    ymin_b = int(max(0, ymin+h_r))
    ymax_t = int(max(0, ymax-h_r)) 
    ymax_b = int(max(0, ymax+h_r))
    xmin_l = int(max(0, xmin-w_r)) 
    xmin_r = int(max(0, xmin+w_r))
    xmax_l = int(max(0, xmax-w_r)) 
    xmax_r = int(max(0, xmax+w_r))

    img = np.copy(image)
    
    right_strip = img[ymin_t:ymax_b, xmax_l:xmax_r, :]
    bottom_strip = img[ymax_t:ymax_b, xmin_l:xmax_r, :]
    left_strip = img[ymin_t:ymax_b, xmin_l:xmin_r, :]
    top_strip = img[ymin_t:ymin_b, xmin_l:xmax_r, :]
    
    return [right_strip, bottom_strip, left_strip, top_strip]
