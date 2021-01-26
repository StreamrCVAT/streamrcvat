import os

path = input() # Accept the input path from the user
print("Path: ",path) # Print the path

isExist = os.path.exists(path) # Check if the path exists

# Defining Rename file convention
file_rename = "frame-"
counter = 1

if isExist:
    files = os.listdir(path) # List all the directories inside the folder
    for file in files:
        print(file, " ", file_rename + str(counter).zfill(3) + '.jpg')
        os.rename(path + "\\" +file, path + "\\" + (file_rename + str(counter).zfill(3) + '.jpg')) # Rename the files
        counter += 1 # Increment the counter



# E:\General\Personal\Pro\Intel Annotation pro\Repo\Annotation\data\frames\dataset1 - 1st group (Brown car)