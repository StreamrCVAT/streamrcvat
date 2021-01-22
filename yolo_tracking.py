import os
import math

#get all path of files
def read_files_path(folder):
	files_path = list()
	for filename in os.listdir(folder):
		files_path.append(os.path.join(folder, filename))
	return files_path


#return centroid pf two points
def centroid(points):
    # ymin xmin ymax xmax
    return [(points[1]+points[3])//2, (points[0]+points[2])//2]

#returnn eucledian distance
def euclid_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

#return the nearest point
def nearest_point(lst_points, centroid_point): #([[1,2,3,4], ..] , [1, 2])
    min_dist = 10000
    min_ind = -1
    for ind, point in enumerate(lst_points):
        dist = euclid_distance(centroid(point), centroid_point)
        if dist < min_dist:
            min_ind = ind
            min_dist = dist
    return lst_points[min_ind]




#initial centroid point of target car
previousCentroid = list()

#get centroids from the user
print("Enter Initial Centroid: ", end="")
previousCentroid = list(map(int, input().split()))

#path to the YOLO output folder
path = 'D:\\smart_space_lab\\Intel_Annotation\\code\\DemoCode\\modelA\\yolo_output_dataset2'
path_to_files_YOLO_output = read_files_path(path)

#iterate through each filename in the YOLO output folder
for filename in path_to_files_YOLO_output:
    # print(filename)

    #open and read contents from the file
    file = open(filename, 'r')
    lines = file.readlines()
    
    for ind, line in enumerate(lines):
        lines[ind] = list(map(int, line.strip().split()[1:]))
    # print(lines)

    current_track_box = nearest_point(lines, previousCentroid)
    print(filename[-7:-4], current_track_box)

    with open(os.getcwd()+"\\yolo_tracked_car_dataset_1\\{}.txt".format(filename[-12:-4]), 'w') as new_file:
        new_file.write('car '+' '.join(map(str, current_track_box))+'\n')
        




    