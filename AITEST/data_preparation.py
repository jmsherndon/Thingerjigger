import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def fetch_yahoo_finance_data(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="2m")
    return data

def load_and_scale_data(symbol):
    data = fetch_yahoo_finance_data(symbol)
    prices = data['Close'].values
    prices = prices.reshape(-1, 1)

    scaler = MinMaxScaler()
    prices = scaler.fit_transform(prices)

    return prices, scaler

# Rest of the functions from the previous data_preparation.py file...


def create_dataset(prices, look_back=30):
    X, Y = [], []
    for i in range(len(prices) - look_back):
        X.append(prices[i:i + look_back])
        Y.append(prices[i + look_back])
    return np.array(X), np.array(Y)

def prepare_data(file_name, look_back=1):
    prices, scaler = load_and_scale_data(file_name)
    X, Y = create_dataset(prices, look_back)
    return X, Y, scaler, prices