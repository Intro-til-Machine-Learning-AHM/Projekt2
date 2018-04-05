import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense

data_x = pd.read_csv("data.csv")
data_y = data_x["class"]
del data_x["class"]

model = Sequential()
model.add(Dense(units=100, activation='tanh', input_dim=8))
model.add(Dense(units=100, activation='tanh'))
model.add(Dense(units=1, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adamax')

model.fit(data_x, data_y, epochs=10, batch_size=10)
