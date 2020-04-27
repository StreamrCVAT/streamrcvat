'''
Model:

-> In this approach there are 4 different models (same structure but different weights)
for 4 edges of the bounding box.

-> Input to Model B - Cropped part of a edge of the bounding box
   Output of Model B - ERROR to be corrected ( eg -4px)

Model training:
1) Train a base model with 32 manually annotated frames for 15 epochs with following metrics
top_model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae'])
top_model.fit(X_train_image[:32,:,:,:], y_train[:32], epochs=15)
->Base model is created

2) Predict for frames, If the prediction is not correct for the given frame
retrain the model with correct I/P  and O/P (only this frame) for 10 epochs.
top_model.fit([[top_part]], [[line2[0]-line1[0]]], epochs=10, callbacks=[es])


Bottom  - 295 / 499
Left    - 280 / 499
Right   - 307 / 499
Top     - 308 / 499


# 1 Convolutional layer
inp = Input(shape=(width,height,depth))
conv1 = Conv2D(32, kernel_size=3, padding='same')(inp)
# batch1 = BatchNormalization()(conv1)
act1 = Activation('relu')(conv1)
pool1 = MaxPooling2D(pool_size=(3,3))(act1)


# 2 Convolutional layer
# conv2 = Conv2D(128, kernel_size=3, padding='same')(pool1)
# act2 = Activation('relu')(conv2)
# pool2 = MaxPooling2D(pool_size=(3,3))(act2)

# Flatten layer
flat2 = Flatten()(pool1)

#Dense Layers
hidden3 = Dense(64)(flat2)
act3 = Activation('relu')(hidden3)
hidden4 = Dense(16)(act3)
act4 = Activation('relu')(hidden4)
output = act4

if(regres == True):
    output_regres = Dense(1, activation='linear')(output)
    output = output_regres
    return Model(inputs=[inp], outputs=[output])
return inp, output

'''