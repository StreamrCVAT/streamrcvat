import tensorflow as tf
#from tensorflow.keras import layers
import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import array
from keras.utils import to_categorical

df = pd.read_csv('E:/General/Personal/Pro/Intel Annotation pro/Repo/Annotation/modelB/yolo_input_rnn.csv')

boundaries = df.iloc[:,1:].values
sequences = []
for i in range(1,len(boundaries)):
    sequence = boundaries[i-1:i+1]
    sequences.append(sequence)

sequences = array(sequences)
X,y = sequences[:,0],sequences[:,1]

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
print(X)
print(y.shape)
model = Sequential()
model.add(LSTM(50, input_shape=(1,4)))
model.add(Dense(4))
print(model.summary())


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=1, verbose=2)

print(model.predict([[[137,161,487,1256]]]))
