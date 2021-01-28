import os
from global_params import *

# Method used to create new folder by checking if it exists before
def createFolder(name):
    folder = os.path.join(ABSOLUTE_PATH, name)
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    print("Folder created: " + folder)

# Fetches the absolute path
ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "\\data"

# Create folders as specified in global_params
createFolder(FRAMES_PATH)
createFolder(YOLO_OUTPUT_PATH)
createFolder(LINEAR_INTERPOL_PATH)
createFolder(MODEL_B_OUTPUT_PATH)
createFolder(FINAL_UI_OUTPUT_PATH)
createFolder(YOLO_OUTPUT_TRACKED_PATH)

# Write the absolute paths of directories paths.txt 
with open(ABSOLUTE_PATH + '\\paths.txt','w') as file:
    file.write(ABSOLUTE_PATH + '\\' + FRAMES_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + YOLO_OUTPUT_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + MODEL_B_OUTPUT_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + LINEAR_INTERPOL_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + FINAL_UI_OUTPUT_PATH + '\n')
    file.write(ABSOLUTE_PATH + '\\' + YOLO_OUTPUT_TRACKED_PATH + '\n')

print("paths.txt file created")