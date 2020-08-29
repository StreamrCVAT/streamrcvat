import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

coors = list()
predicted = list()
skip = 10


def collect_coors():
    #paths to coordinate dirs
    human_annotated_dir = 'D:\\smart_space_lab\\Intel_Annotation\\frames\\annotated frames coor (1 to 499)'
    print('---------------------PATH to DIRS SET---------------------------')
    for filename in os.listdir(human_annotated_dir):
        if(filename=='readme.txt'):
            continue
        print('processing', filename)
        path = os.path.join(human_annotated_dir, filename) #human outputs

        file_data = open(path, 'r')
        line = file_data.readline()
        line = list(map(float, line.split()[1:])) #[ymin xmin ymax xmax]
        print(line)
        coors.append(line)

def y_val(point1, point2, x):
    return (((point2[1]-point1[1])/(point2[0]-point1[0])) * (x-point1[0]) + (point1[1]))


def polatate():
    #[ymin xmin ymax xmax]
    # a     b
    # d     c
    cnt = 0
    l = len(coors)
    
    a = (coors[cnt][1], coors[cnt][0]) # (xmin, ymin)
    b = (coors[cnt][3], coors[cnt][0]) # (xmax, ymin)
    c = (coors[cnt][3], coors[cnt][2]) # (xmax, ymax)
    d = (coors[cnt][1], coors[cnt][2]) # (xmin, ymax)
    predicted.append(coors[cnt])
    cnt = cnt + skip

    while(cnt < l):
        p = (coors[cnt][1], coors[cnt][0]) # (xmin, ymin)
        q = (coors[cnt][3], coors[cnt][0]) # (xmax, ymin)
        r = (coors[cnt][3], coors[cnt][2]) # (xmax, ymax)
        s = (coors[cnt][1], coors[cnt][2]) # (xmin, ymax)

        a_diff = abs(p[0]-a[0]) / (skip-1) # Horizontal Distance between Min(frame1, frame2)
        b_diff = abs(q[0]-b[0]) / (skip-1) # Horizontal Distance between Max(frame1, frame2)
        #[ymin xmin ymax xmax]
        for i in range(1, skip):
            tmp = [
                y_val(a, p, a[0] + a_diff*i), a[0] + a_diff*i, y_val(d, s, a[0] + a_diff*i), b[0] + b_diff*i
            ]
            tmp = list(map(int, tmp))
            predicted.append(tmp)
        predicted.append(coors[cnt])
        a, b, c, d = p, q, r, s
        cnt = cnt + skip

    for i in range(len(coors)-len(predicted)):
        predicted.append(predicted[-1])
    print(predicted)



def visualize():
    right_errors = list()
    for i in range(len(coors)):
        right_errors.append(abs(coors[i][3]-predicted[i][3]))

    total_error = sum(right_errors)
    print("total errors: ", total_error)

collect_coors()
polatate()

with open("top_performance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    top_error = list()
    top_error.append(['top_error'])
    for i in range(len(coors)):
        top_error.append([abs(coors[i][0]-predicted[i][0])])
    writer.writerows(top_error)

with open("right_performance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    right_error = list()
    right_error.append(['right_error'])
    for i in range(len(coors)):
        right_error.append([abs(coors[i][3]-predicted[i][3])])
    writer.writerows(right_error)

with open("bottom_performance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    bottom_error = list()
    bottom_error.append(['bottom_error'])
    for i in range(len(coors)):
        bottom_error.append([abs(coors[i][2]-predicted[i][2])])
    writer.writerows(bottom_error)

with open("left_performance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    left_error = list()
    left_error.append(['left_error'])
    for i in range(len(coors)):
        left_error.append([abs(coors[i][1]-predicted[i][1])])
    writer.writerows(left_error)

print(len(predicted))
print(len(coors))

visualize()
