import os
import math
from global_params import *

# Return centroid pf two points
def centroid(points):
    # ymin xmin ymax xmax
    return [(points[1]+points[3])//2, (points[0]+points[2])//2]

# Return eucledian distance
def euclideanDistance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# Find absolute path
def absolutePathConverter(relativePath):
    ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
    return ABSOLUTE_PATH

# Get the centroid for the Human annotated file
def getFrameCentroid(filePath):
    try:
        with open(filePath, 'r') as file: # VERIFY THE FILE NAME
            frameCoors = file.read()
            frameCentroid = centroid(list(map(int,frameCoors.split()[1:])))
            return frameCentroid
    except:
        print("File not found")
        return    