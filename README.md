# Video Annotation tool

## Overview

The main objective is to build a solution to annotate objects in a video automatically using various methodologies by reducing the overall time and cost consumed in the annotation process.

## Challenges

1) Dataset preparation is a time-consuming process in any machine learning problem and nearly 70% of the total time is used for data preprocessing and annotation.

2) In the Computer Vision arena,  there are a lot of algorithms available to implement the solution. But the tedious task is in collecting relevant images and videos for training the model. Post data collection, the next most expensive task is to annotate the objects in the images and videos for feeding into the model. Though there are pre annotated datasets available online, but in a real-life scenario, the requirements are very different. Thus the whole process of data annotation has to be started from scratch again.

3) A vital requirement for computer vision-based machine learning solutions is to have high-quality data. In the annotation task, there are possibilities for errors due to human bias. For example

>> a)The boundary of the image is not properly marked

>> b)Some objects are missed during the annotation process

>> c)Typically in medical images, the region has to be properly and accurately annotated

>> d)In some scenarios, domain expertise is also required as discussed for the medical image annotation task

4) If annotation to be a manual process, then a lot of human power has to be deployed that requires a huge amount of money to be paid to the workers.

## Dataset and Annotation:

Dataset URL			: https://youtu.be/GkyJmcS2EcA  
Dataset description		: A grey car moving across the camera  
Duration of the video 		: 10 seconds  
Number of frames split		: 500  
Annotation Tool used		: LabelImg  
Annotation Tool URL		: https://github.com/tzutalin/labelImg  
Annotated Image format	: YOLO (Coordinates in .txt file)  
Average time to annotate 1  
Frame				: 12.84 seconds/frame

## Contributors

[Sanjay Tharagesh R S](https://github.com/sanjaytharagesh31)  
[Srishilesh P S](https://github.com/srishilesh)

## Mentors

Dr.Chetan Nategar  
Dr.T.Senthilkumar
