import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def fetch_yahoo_finance_data(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="max")

    # Drop any rows with missing values
    data = data.dropna()

    # Ensure the DataFrame is not empty
    if data.empty:
        return None

    return data


def load_and_scale_data(symbol):
    data = fetch_yahoo_finance_data(symbol)

    # If the data is None, return None for all outputs
    if data is None:
        return None, None

    prices = data['Close'].values
    prices = prices.reshape(-1, 1)

    scaler = MinMaxScaler()
    prices = scaler.fit_transform(prices)

    return prices, scaler

# Rest of the functions from the previous data_preparation.py file...


def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], look_back))  # Ensure X is a 2D array
    return X, np.array(Y)

def prepare_data(symbol, look_back=1):
    prices, scaler = load_and_scale_data(symbol)
    X, Y = create_dataset(prices, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # add the time step dimension
    return X, Y, scaler, prices