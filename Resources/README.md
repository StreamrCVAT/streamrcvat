## Annotation tools

 - https://github.com/heartexlabs/awesome-data-labeling
 
## Dataset

 - https://www.jpjodoin.com/urbantracker/dataset.html

### Folder naming

- frames - contains images based on individual object (car). Eg: Tracking a single brown car
- finalCoordinates - contains the last annotated/correctly predicted coordinates by the human/modelB resp.
- linearInterpolCoordinates - coordinates derived by linear interpolation method
- modelBCoordinates - predicted coordinates based on the output from the deep learning modelB
- yoloCoordinates - contains the yolo coordinates of each frame (number of coordinates based on number of objects)
- yoloTrackedCoordinates - contains a single coordinate of the object that is being tracked/annotated
- paths.txt - contains the absolute paths for all the above directories