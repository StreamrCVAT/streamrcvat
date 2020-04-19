import os
import csv

cwd = os.getcwd()       # Get Current Working directory
path = os.path.join(cwd,"yolo_output")  # Changing to yolo output directory
count = 1   # To count the number of frames
ans = []    # Final list containing all the data
for files in (os.listdir(path)[1:]):    
    file_path = os.path.join(path,files)
    f = open(file_path,"r")
    data = (f.read()).split()
    temp = [count]          # Appending the count of frames
    for j in data[1:]:
        temp.append(int(j)) # Appending the boundary values of the frames
    ans.append(temp)
    count+=1

csv_output_file = os.path.join(cwd,"yolo_input_rnn.csv")

with open(csv_output_file,'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(ans)
print("Data written into CSV file")

