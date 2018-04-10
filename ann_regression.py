import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import os

data_x = pd.read_csv("data.csv")
data_y = data_x["insulin"]
del data_x["class"]
del data_x["insulin"]

model = Sequential()
model.add(Dense(units=200, activation='tanh', input_dim=7))
model.add(Dense(units=100, activation='tanh'))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mse', optimizer='adadelta')

model.fit(data_x, data_y,
          validation_split=0.3, epochs=1000, batch_size=20,
          callbacks = [EarlyStopping(monitor='val_loss', patience=20)],
          verbose = 1)
