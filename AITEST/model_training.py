import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

def create_model(look_back=1):
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))  # changed input_shape
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, X_train, Y_train, epochs=25, batch_size=1):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model