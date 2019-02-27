from numpy import array
from scipy.io import loadmat
import numpy as np  
import pandas as pd

from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


x = loadmat('/Users/sanjana/desktop/CG-Mini-Project/Data/M1-DeepSquat.mat')
arr = x['Train_Data']
arr_list = []
for i in range(0,90):
    arr_list.append(arr[0,i])
arr = np.dstack(arr_list)
arr = np.rollaxis(arr,-1)

sequence = arr[1,:,:]
sequence = sequence.reshape((1,240,117))

samples = 90
features = 117
timesteps = 240

model = Sequential()

model.add(LSTM(30, activation='relu', input_shape=(timesteps,features)))
model.add(RepeatVector(timesteps))

model.add(LSTM(10, activation='relu'))
model.add(RepeatVector(timesteps))

model.add(LSTM(4, activation='relu'))
model.add(RepeatVector(timesteps))

model.add(LSTM(10, activation='relu'))
model.add(RepeatVector(timesteps))

model.add(LSTM(30, activation='relu'))
model.add(RepeatVector(timesteps))

model.add(LSTM(117, activation='relu', return_sequences=True))

model.compile(optimizer='adam', loss='mse')
model.fit(sequence, sequence, epochs=300, verbose=0)

model = Model(inputs=model.inputs, outputs=model.layers[2].output)

yhat = model.predict(sequence)
print(yhat.shape)
print(sequence.shape)
print(yhat)
