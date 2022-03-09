import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM



import numpy as np
import matplotlib.pyplot as plt
import requests

API_KEY = "KW8CD0CRFLR2LKS0"


def get_stock_price(symbol: str):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo'
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "apikey": API_KEY}
    try:
        r = requests.get(url, params=params)
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    data = r.json()
    return data["Time Series (Daily)"]


def plot_stock_trend(x_data: list, y_data: list, symbol: str):
    plt.title("Symbol " + str(symbol))
    # plt.ylabel('')
    # plt.xlabel('')
    plt.plot(x_data, y_data)
    plt.show()


def split_stock_prices_data(dataset, N, offset):
    pass

def create_dataset(dataset):
    data_x = [], data_y = []
    for i in range(dataset):
        print("test")

    return np.array(data_x), np.array(data_y)



def main():
    # Stock price prediction using LSTM
    # Load the data
    symbol = "AAPL"
    data = get_stock_price(symbol)
    plot_stock_trend(data.keys(), data.keys(), symbol)
    # Train and test split
    test_ratio = 0.2
    training_ratio = 1 - test_ratio
    split_stock_prices_data(data, training_ratio, 0)

    # Data preprocessing
    data_test = []
    x_train, y_train = create_dataset(data)
    x_test, y_test = create_dataset(data_test)

    # LSTM
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Sequential model


    # Prediction

    # Conclusion


if __name__ == "__main__":
    main()
