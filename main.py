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
        if (len(os.listdir(finalPath)) == batchSize):
            return os.listdir(finalPath)[-1]   

def main():
    # firstCentroid = getFrameCentroid("frame-001.txt")
    try:
        batchLastFileName = alertFrame32()
        firstCentroid = helper.getFrameCentroid(batchLastFileName, FINAL_UI_OUTPUT_PATH)
        yoloTracker.trackObject(firstCentroid) # Enable live YOLO tracker for the object
        print("YOLO Tracker completed!")
    except:
        print("")

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