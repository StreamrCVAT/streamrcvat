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

def draw_line(image, start_point, end_point):
    # start_point = (0, 0) 
    # end_point = (250, 250) 
    start_point = tuple(map(int, start_point))
    end_point = tuple(map(int, end_point))
    img = np.copy(image)
    color = (0, 255, 0) 
    thickness = 2
    image = cv2.line(img, start_point, end_point, color, thickness) 
    return image

def draw_rect(image, start_point, end_point):
    # start_point = (xmin, ymin) 
    # end_point = (xmax, ymax)
    img = np.copy(image)
    
    color = (255, 0, 0) 
    thickness = 2
    annotated = cv2.rectangle(img, start_point, end_point, color, thickness)
    return annotated

def strip(image, xmin, ymin, xmax, ymax, line=False):
    #return strip of the 4 edges
    #image is 3D numpy array, coordinates are output of YoLo model
    frame_w = image.shape[1]
    frame_h = image.shape[0]

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
    img = np.copy(image)
    if(line == True):
        img = draw_line(image, (xmax,0), (xmax,frame_h))
        right_strip = img[ymin_t:ymax_b, xmax_l:xmax_r, :]

        img = draw_line(image, (0,ymax), (frame_w,ymax))
        bottom_strip = img[ymax_t:ymax_b, xmin_l:xmax_r, :]

        img = draw_line(image, (xmin,0), (xmin,frame_h))
        left_strip = img[ymin_t:ymax_b, xmin_l:xmin_r, :]

        img = draw_line(image, (0,ymin), (frame_w, ymin))
        top_strip = img[ymin_t:ymin_b, xmin_l:xmax_r, :]
        
        return [right_strip, bottom_strip, left_strip, top_strip]
    else:
        right_strip = img[ymin_t:ymax_b, xmax_l:xmax_r, :]
        bottom_strip = img[ymax_t:ymax_b, xmin_l:xmax_r, :]
        left_strip = img[ymin_t:ymax_b, xmin_l:xmin_r, :]
        top_strip = img[ymin_t:ymin_b, xmin_l:xmax_r, :]
        
        return [right_strip, bottom_strip, left_strip, top_strip]

def visualize_strips(image, xmin, ymin, xmax, ymax):
    print('image shape', (image).shape) # (Y, X, DEPTH=3)

    start_point = (xmin, ymin) 
    end_point = (xmax, ymax)
    annotated = draw_rect(image, start_point, end_point)

    cv2.imshow('YOLO Image',annotated)   
    cv2.waitKey(0)
    # car 155 624 320 960
    ret = strip(image, 624, 155, 960, 320, True)

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
