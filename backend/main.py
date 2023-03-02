import os.path
from datetime import date, datetime

import keras
import json
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import csv
import requests
import math
from sklearn.preprocessing import MinMaxScaler
from keras import backend

API_KEY = "KW8CD0CRFLR2LKS0"


def get_stock_price(symbol, new=False):
    dataset = {}
    file_name = 'stock_' + str(symbol) + '.csv'
    file_path = 'backend/data/' + file_name

    if os.path.exists(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset[row["date"]] = float(row["close_price"])
    else:
        url = 'https://www.alphavantage.co/query'
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "apikey": API_KEY, "outputsize": "full"}
        try:
            r = requests.get(url, params=params)
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)

        data = r.json()
        for key, value in data["Time Series (Daily)"].items():
            dataset[key] = value["4. close"]

        csv_columns = ["date", "close_price"]
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for key, value in dataset.items():
                writer.writerow({"date": key, "close_price": value})

    return dataset


def create_dataset(dataset, n_days):
    data_x = []
    data_y = []
    for key, values in dataset.items():
        data_x.append(key)
        data_y.append(values)

    data_x.reverse()
    data_y.reverse()
    data_x = data_x[-n_days:]
    data_y = data_y[-n_days:]
    return np.array(data_x), np.array(data_y)


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_loss_chart(history_dict, param):
    loss_values = history_dict['loss']
    epochs = range(1, len(loss_values) + 1)
    mape_values = history_dict['mean_absolute_percentage_error']
    rmse_values = history_dict['root_mean_squared_error']
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, mape_values, 'b', label='MAPE')
    plt.plot(epochs, rmse_values, 'r', label='RMSE')

    plt.title('Training loss ' + param)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def stock_prediction_lstm(symbol: str = "AAPL", n_days: int = 365, plot: bool = False, new_model: bool = False):
    #####################################
    # Stock price prediction using LSTM #
    #####################################
    # Author: Juan Llamazares Ruiz

    ###################
    # Data extraction #
    ###################

    data = get_stock_price(symbol)
    # Train and test split
    test_ratio = 0.2
    training_ratio = 1 - test_ratio
    train_size = int(training_ratio * n_days)
    test_size = int(test_ratio * n_days)
    print("train_size: " + str(train_size))
    print("test_size: " + str(test_size))
    ######################
    # Data preprocessing #
    ######################
    date_list, price_list = create_dataset(data, n_days)
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_list.reshape(-1, 1))
    # Create training and test data
    train = scaled_data[0:train_size]
    test = scaled_data[train_size - test_size:train_size]
    ##################################
    # Build and train the LSTM model #
    ##################################
    metrics = [
        keras.metrics.RootMeanSquaredError(),  # RMSE
        keras.metrics.MeanAbsolutePercentageError(),  # MAPE

    ]

    history_lstm = history_gru = history_dict_lstm = history_dict_gru = None
    try:
        if new_model:
            lstm_model, history_lstm = create_lstm_model(train, metrics)
            lstm_model.save("models/lstm_model_" + symbol + ".h5")
            history_dict_lstm = history_lstm.history
            json.dump(history_dict_lstm, open("models/lstm_model_" + symbol + "_history.json", "w"))

            gru_model, history_gru = create_gru_model(train, metrics)
            gru_model.save("models/gru_model_" + symbol + ".h5")
            history_dict_gru = history_gru.history
            json.dump(history_dict_gru, open("models/gru_model_" + symbol + "_history.json", "w"))
        else:
            try:
                lstm_model = keras.models.load_model("models/lstm_model_" + symbol + ".h5")
                history_dict_lstm = json.load(open("models/lstm_model_" + symbol + "_history.json", "r"))
            except OSError:
                lstm_model, history = create_lstm_model(train, metrics)
                lstm_model.save("models/lstm_model_" + symbol + ".h5")
                history_dict_lstm = history.history
                json.dump(history_dict_lstm, open("models/lstm_model_" + symbol + "_history.json", "w"))

    except:
        print("[ERROR] Creating/Reading model")

    ###################
    # Plot loss chart #
    # #################
    if plot:
        if history_dict_lstm is not None:
            plot_loss_chart(history_dict_lstm, "LSTM")

        if history_dict_gru is not None:
            plot_loss_chart(history_dict_gru, "GRU")

    #############################
    # Make predictions and test #
    #############################
    train_predict = lstm_model.predict(train)
    test_predict = lstm_model.predict(test)

    #########################
    # Calculate the metrics #
    # #########################
    # val_rmse = lstm_model.history[]

    rmse_train = rmse(train, train_predict)
    rmse_test = rmse(test, test_predict)
    mape_train = mape(train, train_predict)
    mape_test = mape(test, test_predict)
    print("RMSE train: " + str(rmse_train))
    print("RMSE test: " + str(rmse_test))
    print("MAPE train: " + str(mape_train))
    print("MAPE test: " + str(mape_test))

    # # Save metrics
    # metrics_dict = {
    #     "rmse_train": rmse_train,
    #     "rmse_test": rmse_test,
    #     "mape_train": mape_train,
    #     "mape_test": mape_test
    # }
    # date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    # json.dump(metrics_dict, open("models/lstm_model_" + symbol + "_metrics.json", "w"))
    #

    # Invert scaling for forecast
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    test_predict = test_predict[::-1]
    test = scaler.inverse_transform(test)
    #############################
    # Plot the results #
    #############################
    if plot:
        plot_predicted_data(test_predict, train_predict, date_list, price_list, symbol)
        plot_metric_results(rmse_train, rmse_test, mape_train, mape_test, symbol)

    ####################
    # Save the results #
    ####################
    lstm_model.summary()

    ret = lstm_model.evaluate(test, test_predict)
    # print(ret)

    predicted_data = np.concatenate((train_predict, test_predict), axis=0)
    predicted_data = predicted_data.tolist()

    results = {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "mape_train": mape_train,
        "mape_test": mape_test,
        "model": lstm_model,
        "predicted_data": predicted_data,
    }

    return results


def plot_predicted_data(test_predict, train_predict, date_list, price_list, symbol):
    plt.plot(date_list, price_list, color='blue', label='Real stock price data')
    plt.plot(date_list[:len(train_predict)], train_predict.flatten(), color='green', label='Train')
    plt.plot(date_list[len(train_predict):], test_predict.flatten(), color='black', label='Prediction')
    plt.title("Stock Price Prediction")
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.xticks(np.arange(0, len(date_list), 50), date_list[::50], rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend(loc='best')
    file_name = "RESULT_" + datetime.today().strftime('%d%m%Y_%H:%M') + "_" + symbol
    if not os.path.exists("results/" + file_name + ".png"):
        plt.savefig("results/" + file_name + ".png")
    plt.show()


def plot_metric_results(rmse_train, rmse_test, mape_train, mape_test, symbol, save=False):
    plt.plot(rmse_train, color='red', label='Train')
    plt.plot(rmse_test, color='black', label='Test')
    plt.title("RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend(loc='best')
    if save:
        file_name = "RMSE_" + datetime.today().strftime('%d%m%Y_%H:%M') + "_" + symbol
        if not os.path.exists("results/" + file_name + ".png"):
            plt.savefig("results/" + file_name + ".png")
        # else:
        #     plt.savefig("results/"+file_name+"_.png")
    plt.show()

    plt.plot(mape_train, color='red', label='Train')
    plt.plot(mape_test, color='black', label='Test')
    plt.title("MAPE")
    plt.xlabel("Epoch")
    plt.ylabel("MAPE")
    plt.legend(loc='best')
    if save:
        file_name = "MAPE_" + datetime.today().strftime('%d%m%Y_%H:%M') + "_" + symbol
        plt.savefig("results/" + file_name + ".png")
    plt.show()

    if save:
        file_name = "NUMERIC_" + datetime.today().strftime('%d%m%Y_%H:%M') + "_" + symbol
        f = open("results/" + file_name + ".txt", "w")
        f.write("RMSE train: " + str(rmse_train))
        f.write("RMSE test: " + str(rmse_test))
        f.write("MAPE train: " + str(mape_train))
        f.write("MAPE test: " + str(mape_test))
        f.close()


def create_lstm_model(train, metrics):
    lstm_model = Sequential()
    # First layer
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(train.shape[1], 1)))
    # Second layer
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    # Compiling the RNN (Recurrent Neuronal Network)
    lstm_model.compile(optimizer="adam", loss='mse', metrics=metrics)
    # Fitting the RNN to the Training set
    history = lstm_model.fit(train, train, epochs=100, batch_size=32, verbose=2)
    return lstm_model, history


def create_gru_model(train, metrics):
    gru_model = Sequential()
    # First layer
    gru_model.add(GRU(units=50, return_sequences=True, input_shape=(train.shape[1], 1)))
    # Second layer
    gru_model.add(GRU(units=50))
    gru_model.add(Dense(1))
    # Compiling the RNN (Recurrent Neuronal Network)
    gru_model.compile(optimizer="adam", loss='mse', metrics=metrics)
    # Fitting the RNN to the Training set
    history = gru_model.fit(train, train, epochs=100, batch_size=32, verbose=2)
    return gru_model, history


def generate_all_predictions():
    n_days = 365
    stock_list = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL']
    for stock in stock_list:
        stock_prediction_lstm(plot=False, new_model=True, symbol=stock, n_days=n_days)


def main():
    stock_prediction_lstm(plot=True, new_model=True)
    # generate_all_predictions()


if __name__ == "__main__":
    main()
