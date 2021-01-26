import os
from global_params import *

def createFolder(name):
    folder = os.path.join(ABSOLUTE_PATH, name)
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    print("Folder created: " + folder)

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "\\data"

createFolder('frames')
createFolder('yoloCoordinates')
createFolder('linearInterpolCoordinates')
createFolder('modelBCoordinates')
createFolder('finalCoordinates')


with open(ABSOLUTE_PATH + '\\paths.txt','w') as file:
    file.write(ABSOLUTE_PATH + '\\' + FRAMES_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + YOLO_OUTPUT_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + MODEL_B_OUTPUT_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + LINEAR_INTERPOL_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + FINAL_UI_OUTPUT_PATH + '\n')

print("paths.txt file created")