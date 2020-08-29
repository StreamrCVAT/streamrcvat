import os

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

def y_val(a, b, x):
    return (((b[1]-a[1])/(b[0]-a[0])) * (x-a[0]) + (a[1]))


def polatate():
    #[ymin xmin ymax xmax]
    # a     b
    # d     c
    cnt = 0
    l = len(coors)
    a = (coors[cnt][1], coors[cnt][0])
    b = (coors[cnt][1], coors[cnt][2])
    c = (coors[cnt][3], coors[cnt][2])
    d = (coors[cnt][1], coors[cnt][2])
    predicted.append(coors[cnt])
    cnt = cnt + skip

    while(cnt < l):
        p = (coors[cnt][1], coors[cnt][0])
        q = (coors[cnt][1], coors[cnt][2])
        r = (coors[cnt][3], coors[cnt][2])
        s = (coors[cnt][1], coors[cnt][2])

        a_diff = p[0]-a[0] / (skip-1)
        b_diff = q[0]-b[0] / (skip-1)
        #[ymin xmin ymax xmax]
        for i in range(1, skip):
            predicted.append([
                y_val(a, p, a[0] + a_diff*i), a[0] + a_diff*i, y_val(d, s, a[0] + a_diff*i), b[0] + b_diff*i
            ])
        predicted.append(coors[cnt])
        a, b, c, d = p, q, r, s
        cnt = cnt + skip

    print(predicted)




collect_coors()
polatate()


print(len(predicted))
print(len(coors))

# (x1, y2)

#                 (x2, y2)


#     y2-y1   y-y1
#     x2-x1 = x-x1

# y = ((y2-y1) / (x2-x1) ) * (x-x1) + y1


