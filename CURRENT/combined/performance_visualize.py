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

def visualize_performance(performance):
    total_yolo_error = 0.0
    total_modelB_error = 0.0
    yolo_errors = list()
    modelB_errors = list()
    print(performance[0])
    for i in performance[1:]:
        print(i)
        total_modelB_error += i[4]
        total_yolo_error += i[3]
        yolo_errors.append(i[3])
        modelB_errors.append(i[4])
    print('Sum of YOLO errors', total_yolo_error)
    print('Sum of ModelB errors', total_modelB_error)
    
    # print(yolo_errors)
    # print(modelB_errors)
    plt.plot(yolo_errors, label = 'yolo error')
    plt.plot(modelB_errors, label = 'modelB error')
    plt.ylabel('error')
    plt.xlabel('frame #')
    plt.legend()
    plt.show()

    plt.bar(['SUM(YOLOerrors)', 'SUM(ModelBerrors)'], [total_yolo_error, total_modelB_error])
    plt.show()

    plt.hist(yolo_errors)
    plt.xlabel('Error range')
    plt.ylabel('Frame count')
    plt.show()

    plt.hist(modelB_errors)
    plt.xlabel('Error range')
    plt.ylabel('Frame count')
    plt.show()
