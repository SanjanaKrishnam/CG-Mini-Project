from numpy import array
from scipy.io import loadmat
import numpy as np  
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

x = loadmat('/Users/sanjana/desktop/M1-DeepSquat.mat')
arr = x['Train_Data']
arr_list = []
for i in range(0,90):
    arr_list.append(arr[0,i])
arr = np.dstack(arr_list)
arr = np.rollaxis(arr,-1)

samples = 90
features = 117
timesteps = 240


model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(timesteps,features)))
model.add(RepeatVector(timesteps))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)
# connect the encoder LSTM as the output layer
model = Model(inputs=model.inputs, outputs=model.layers[0].output)
plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
# get the feature vector for the input sequence
yhat = model.predict(sequence)
print(yhat.shape)
print(yhat)