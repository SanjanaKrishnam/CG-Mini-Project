from numpy import array
from scipy.io import loadmat
import numpy as np  
import pandas as pd

from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed


x = loadmat('/Users/sanjana/desktop/CG-Mini-Project/Data/M1-DeepSquat.mat')
arr = x['Train_Data']
arr_list = []
for i in range(0,90):
    arr_list.append(arr[0,i])
arr = np.dstack(arr_list)
arr = np.rollaxis(arr,-1)

samples = arr.shape[0]
timesteps = arr.shape[1]
features = arr.shape[2]

model = Sequential()

model.add(LSTM(30, activation='relu', return_sequences = True, input_shape=(timesteps,features)))
model.add(LSTM(10, activation='relu', return_sequences = True))
model.add(LSTM(4, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences = True))
model.add(LSTM(30, activation='relu', return_sequences = True))
model.add(LSTM(117, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(features), input_shape=(timesteps,features)))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(arr, arr, validation_split=0.33, epochs = 300)

model = Model(inputs=model.inputs, outputs=model.layers[2].output)

Train_Data_Reduced = model.predict(arr)










