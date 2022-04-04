import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import requests
from sklearn.preprocessing import MinMaxScaler

API_KEY = "KW8CD0CRFLR2LKS0"


def get_stock_price(symbol: str):
    url = 'https://www.alphavantage.co/query'
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "apikey": API_KEY, "outputsize": "compact"}
    try:
        r = requests.get(url, params=params)
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    data = r.json()
    return data["Time Series (Daily)"]


def store_stock_price(dataset, symbol):
    csv_columns = ["date", "close_price"]
    with open(str(symbol) + '.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dataset.values():
            writer.writerow(data)


def read_stock_price(symbol):
    dataset = {}
    with open(str(symbol) + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # pass header
        try:
            for row in reader:
                dataset[row[0]] = {"4. close": float(row[1])}
        except ValueError:
            print(row)
    return dataset


def plot_stock_trend(x_data: list, y_data: list, symbol: str):
    plt.title("Symbol " + str(symbol))
    # plt.ylabel('')
    # plt.xlabel('')
    plt.plot(x_data, y_data)
    plt.show()


def create_dataset(dataset):
    data_x = []
    data_y = []
    for key, values in dataset.items():
        data_x.append(key)
        data_y.append(values["4. close"])

    return np.array(data_x), np.array(data_y)


def main():
    #####################################
    # Stock price prediction using LSTM #
    #####################################
    # Author: Juan Llamazares

    ###################
    # Data extraction #
    ###################
    symbol = "AAPL"
    # data = get_stock_price(symbol)
    # test = {}
    # idx = 0
    # for key, value in data.items():
    #     test[idx] = {"date": key, "close_price": value["4. close"]}
    #     idx += 1
    #
    # store_stock_price(test, symbol)
    data = read_stock_price(symbol)

    # Train and test split
    test_ratio = 0.2
    training_ratio = 1 - test_ratio
    train_size = int(training_ratio * len(data))
    test_size = int(test_ratio * len(data))
    print("train_size: " + str(train_size))
    print("test_size: " + str(test_size))

    ######################
    # Data preprocessing #
    ######################
    x, y = create_dataset(data)

    new_data = pd.DataFrame(index=range(0, len(data)), columns=['date', 'close'])

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.reshape(y))

    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.2)

    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    ##################################
    # Build and train the LSTM model #
    ##################################
    lstm_model = Sequential()
    # First layer
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train, 1)))

    # Second layer
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    metrics = [
        keras.metrics.RootMeanSquaredError(name="oot_mean_squared_error"),  # RMSE
        keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error"),  # MAPE
    ]

    # Compiling the RNN (Recurrent Neuronal Network)
    lstm_model.compile(optimizer="adam", loss='mean_squared_error', metrics=metrics)

    # Fitting the RNN to the Training set
    lstm_model.fit(x_train, y_train, epochs=100, batch_size=32)


    ####################
    # Make predictions #
    ####################
    prediction_stock_price = lstm_model.predict(x_test)
    # prediction_stock_price = scaler.inverse_transform(prediction_stock_price)

    #####################
    # Visualize results #
    #####################

    plot_stock_trend(x_train, prediction_stock_price, symbol)

    # Matriz de confusion
    # print(confusion_matrix(y_val, y_pred))

    # Observa la medida macro-f1 del siguiente informe
    # print(classification_report(y_val, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
