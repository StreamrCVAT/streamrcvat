import os
from global_params import *

# Method used to create new folder by checking if it exists before
def createFolder(name):
    folder = os.path.join(WORKDIR, name)
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    print("Folder created: " + folder)

# Fetches the absolute path
WORKDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'workdir')

# Create folders as specified in global_params
createFolder(FRAMES_PATH)
createFolder(YOLO_OUTPUT_PATH)
createFolder(LINEAR_INTERPOL_PATH)
createFolder(MODEL_B_OUTPUT_PATH)
createFolder(FINAL_UI_OUTPUT_PATH)
createFolder(YOLO_OUTPUT_TRACKED_PATH)

# Write the absolute paths of directories paths.txt 
with open(os.path.join(WORKDIR, 'paths.txt'),'w') as file:
    file.write(WORKDIR + '/' + FRAMES_PATH + '\n')
    file.write(WORKDIR + '/' + YOLO_OUTPUT_TRACKED_PATH + '\n')
    file.write(WORKDIR + '/' + MODEL_B_OUTPUT_PATH + '\n')
    file.write(WORKDIR + '/' + LINEAR_INTERPOL_PATH + '\n')
    file.write(WORKDIR + '/' + FINAL_UI_OUTPUT_PATH + '\n')
    file.write(WORKDIR + '/' + YOLO_OUTPUT_PATH + '\n')

print("paths.txt file created")
print("Please ensure that you paste the dataset frames into the '/frames' folder")

# cp -r Sample_workdir/frames/dataset2\ -\ 1st\ group\ \(White\ car1\)/* workdir/frames/