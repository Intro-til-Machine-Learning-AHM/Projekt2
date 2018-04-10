import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import os

data_x = pd.read_csv("data.csv")
data_y = data_x["insulin"]
del data_x["class"]
del data_x["insulin"]

def train(data_x, data_y):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(7,)))
    model.add(Dense(units=200, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(units=100, activation='tanh'))
    model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mse', optimizer='adamax')

    model.fit(data_x, data_y,
              validation_split=0.3, epochs=1000, batch_size=10,
              callbacks = [EarlyStopping(monitor='val_loss', patience=10)],
              verbose = 0)
    return (model.predict, model)
