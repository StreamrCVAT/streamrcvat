import os
import math
from global_params import *
from utils import helper

# Get all path of files
def getFilesPathAsList(folder):
	files_path = list()
	for filename in os.listdir(folder):
		files_path.append(os.path.join(folder, filename))
	return files_path

# Return the nearest point
def objectInNextFrame(listOfCoordinates, centroid_point): #([[1,2,3,4], [1,2,3,4] ..] , [1, 2])
    min_dist = 10000
    min_ind = -1
    for ind, point in enumerate(listOfCoordinates):
        dist = helper.euclideanDistance(helper.centroid(point), centroid_point)
        if dist < min_dist:
            min_ind = ind
            min_dist = dist
    return listOfCoordinates[min_ind]

# Track the object
def trackNextObject(frame_number, batchLastFileName):

    finalPath = ABSOLUTE_PATH + "\\data\\" + FINAL_UI_OUTPUT_PATH
    batchLastFileName = finalPath   + "\\" + batchLastFileName

    # Path to the YOLO output folder
    yoloOutputPath = ABSOLUTE_PATH + "\\data\\" + YOLO_OUTPUT_PATH
    yoloOutputPathFile = getFilesPathAsList(yoloOutputPath)[int(frame_number)]
    previousObjectCentroid = helper.getFrameCentroid(batchLastFileName)
    # Open and read contents from the file
    file = open(yoloOutputPathFile, 'r')
    lines = file.readlines()
    
    for ind, line in enumerate(lines):
        lines[ind] = list(map(int, line.strip().split()[1:]))
    # print(lines)

    objectInCurrentFrame = objectInNextFrame(lines, previousObjectCentroid)
    print(yoloOutputPathFile[-7:-4], objectInCurrentFrame)

    new_file_directory = ABSOLUTE_PATH + "\\data\\" + YOLO_OUTPUT_TRACKED_PATH + "\\{}.txt".format(yoloOutputPathFile[-12:-4])
    with open(new_file_directory, 'w') as new_file:
        new_file.write('car '+' '.join(map(str, objectInCurrentFrame))+'\n')
    return new_file_directory

# Track the object
def trackObject(previousObjectCentroid):
    # Path to the YOLO output folder
    yoloOutputPath = ABSOLUTE_PATH + "\\data\\" + YOLO_OUTPUT_PATH
    yoloOutputPathFiles = getFilesPathAsList(yoloOutputPath)
    
    # Iterate through each filename in the YOLO output folder - start tracking from batchSize(32)
    for yoloOutputPathFile in yoloOutputPathFiles:

        # Open and read contents from the file
        file = open(yoloOutputPathFile, 'r')
        lines = file.readlines()
        
        for ind, line in enumerate(lines):
            lines[ind] = list(map(int, line.strip().split()[1:]))
        # print(lines)

        objectInCurrentFrame = objectInNextFrame(lines, previousObjectCentroid)
        previousObjectCentroid = objectInCurrentFrame
        print(yoloOutputPathFile[-7:-4], objectInCurrentFrame)

        with open(ABSOLUTE_PATH + "\\data\\" + YOLO_OUTPUT_TRACKED_PATH + "\\{}.txt".format(yoloOutputPathFile[-12:-4]), 'w') as new_file:
            new_file.write('car '+' '.join(map(str, objectInCurrentFrame))+'\n')

# Accept the centroid via CLI
# def getObjectCentroidCLI():
#     # Initialize centroid point of target car
#     previousObjectCentroid = list()
#     # Get centroids from the user
#     print("Enter Initial Centroid: ", end="")
#     previousObjectCentroid = list(map(int, input().split()))
#     trackObject(previousObjectCentroid)

# Accept the centroid via args
def getObjectCentroid(previousObjectCentroid):
    trackObject(previousObjectCentroid)

def main():
    pass

if __name__ == '__main__':
    main()