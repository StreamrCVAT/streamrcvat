'''
Model:

-> In this approach there are 4 different models (same structure but different weights)
for 4 edges of the bounding box.

-> Input to Model B - [Cropped part of a edge of the bounding box] and [Absolute value of Yolo output coordinate]
   Output of Model B - ERROR to be corrected ( eg -4px)

Model training:
1) Train a base model with 32 manually annotated frames for 15 epochs with following metrics
top_model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])
top_model.fit(X_train_image[:32,:,:,:], y_train[:32], epochs=15)
->Base model is created

2) Predict for frames, If the prediction is not correct for the given frame
retrain the model with correct I/P  and O/P (only this frame) for 10 epochs.
top_model.fit([[top_part]], [[line2[0]-line1[0]]], epochs=10, callbacks=[es])


Bottom  - 298 / 499
Left    - 280 / 499
Right   - 303 / 499
Top     - 305 / 499

'''