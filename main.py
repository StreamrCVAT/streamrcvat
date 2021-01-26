import os
from global_params import *
from utils.yoloTracker import *

def alertFrame32():
    finalPath = os.path.dirname(os.path.realpath(__file__)) + "\\data" + FINAL_UI_OUTPUT_PATH
    while(True):
        if (len(os.listdir(finalPath)) > 32):
            break
    return True


