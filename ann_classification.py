import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

def train(data_x, data_y):
    data_y = np.array([data_y, 1 - data_y]).T
    model = Sequential()
    model.add(BatchNormalization(input_shape=(8,)))
    model.add(Dense(units=500, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(units=100, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss=keras.losses.kullback_leibler_divergence, optimizer='adadelta', metrics=['acc'])

    model.fit(data_x, data_y,
              validation_split=0.3, epochs=1000, batch_size=10,
              callbacks = [EarlyStopping(monitor='val_acc', patience=10)],
              verbose = 0)

    def predict(data):
        return [1 if y > 0.5 else 0 for y in model.predict(data)[:,0]]
    return (predict, model)
