import os
import sys
sys.path.insert(0, '/utils/')

from global_params import *
from utils import yoloTracker
from utils import helper

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))


# To alert when annotation for 32 frames are over
def alertFrame32():
    finalPath = os.path.dirname(os.path.realpath(__file__)) + "\\data" + FINAL_UI_OUTPUT_PATH
    while(True):
        if (len(os.listdir(finalPath)) > 32):
            break
    return True   

# Get 
def getFirstFrameCentroid():
    firstFramePath = ABSOLUTE_PATH + "\\data\\" + FINAL_UI_OUTPUT_PATH + "\\frame-001.txt"
    try:
        with open(firstFramePath, 'r') as firstFrame: # VERIFY THE FILE NAME
            firstFrameCoors = firstFrame.read()
            firstFrameCentroid = helper.centroid(list(firstFrameCoors.split()))
            return firstFrameCentroid
    except:
        print("File not found")
        return

def main():
    getFirstFrameCentroid()

if __name__ == '__main__':
    main()




# def main():
#     whilr(True):
#         //check if frame 1 is there int eh dir
#         //track yolo 
#         //calculate error from yolo and human - to traint he base model (Store them in a list)
#         left_error = [] ->append to the list
#         right_error = []
#         top_error = []
#         bot_error = []
#         if(frame == 32):
#             create_train_base_model(erroe, frames path): ->right, left, top, bottom    