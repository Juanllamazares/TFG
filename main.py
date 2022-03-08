import pandas as pd
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


def generate_graph(x_data: list, y_data: list, symbol: str):
    plt.title("Symbol " + str(symbol))
    # plt.ylabel('')
    # plt.xlabel('')
    plt.plot(x_data, y_data)
    plt.show()


def main():
    # Stock price prediction using LSTM
    # Load the data
    symbol = "AAPL"
    data = get_stock_price(symbol)
    generate_graph(data.keys(), data.keys(), symbol)
    # Train and test split

    # Data preprocessing

    # LSTM

    # Prediction

    # Conclusion


if __name__ == "__main__":
    main()
