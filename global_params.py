import os

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
# path to dataset image frames
FRAMES_PATH = 'frames'
YOLO_OUTPUT_TRACKED_PATH = 'yoloTrackedCoordinates'
YOLO_OUTPUT_PATH = 'yoloCoordinates'
LINEAR_INTERPOL_PATH = 'linearInterpolCoordinates'
MODEL_B_OUTPUT_PATH = 'modelBCoordinates'
FINAL_UI_OUTPUT_PATH = 'finalCoordinates'

BATCH_SIZE = 32

FRAME_HH = 720
FRAME_WW = 1280

MODEL_HEIGHT = FRAME_HH//6
MODEL_WIDTH =  FRAME_WW//10

STRIP_HEIGHT = FRAME_HH//2
STRIP_WIDTH =  FRAME_WW//2


# Dataset file format conventions
# FRAMES_PATH = 'frames\\dataset2 - 1st group (White car1)'
# YOLO_OUTPUT_TRACKED_PATH = 'yoloTrackedCoordinates'
# YOLO_OUTPUT_PATH = 'yoloCoordinates'
# LINEAR_INTERPOL_PATH = 'linearInterpolCoordinates'
# MODEL_B_OUTPUT_PATH = 'modelBCoordinates'
# FINAL_UI_OUTPUT_PATH = 'finalCoordinates'
