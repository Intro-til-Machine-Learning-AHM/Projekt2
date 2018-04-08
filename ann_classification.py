import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

data_x = pd.read_csv("data.csv")
data_y = data_x["class"]
data_y = np.array([data_y, 1 - data_y]).T
del data_x["class"]

model = Sequential()
model.add(Dense(units=500, activation='tanh', input_dim=8))
model.add(Dropout(0.5))
model.add(Dense(units=100, activation='tanh'))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss=keras.losses.kullback_leibler_divergence, optimizer='adamax', metrics=['acc'])

model.fit(data_x, data_y, validation_split=0.3, epochs=1000, batch_size=1000)
