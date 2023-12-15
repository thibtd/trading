import numpy as np
import csv
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import date
import datetime
import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import CSVLogger
import streamlit as st
import plotly.graph_objects as go
from tqdm.keras import TqdmCallback
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)


class lstm:
    """
    Description
    ---
    A class for the LSTM object.

    Attributes
    ---
    ticker (str): name of the stock
    data: the Closing price of the stock
    n_lookback (int): the number of days to lookback into
    n_forecast (int): the number of days to predict

    Methods
    ---
    make_model: creates the structure of the model
    make_sequences: creates sequences from the data
    preprocess_lstm: preprosess the data
    make_train_X:
    train: train the model on training data
    forecast: use the trained model to predict the next n days
    save_model: save the model to file
    bool_existing: check whether the model has been trained and saved
    load_model: load a saved model
    plot_training: plot the training error
    plots: plot the historical and estimated data

    """

    def __init__(self, ticker, data, n_lookback, n_forecast):
        self.ticker = ticker
        self.data = data
        self.n_lookback = n_lookback
        self.n_forecast = n_forecast
        self.model = self.make_model()

    def make_model(self):
        if self.bool_existing():
            print('loading model')
            self.trained = True
            model = self.load_model()
            self.history = pd.read_csv('trainings/training{}.log'.format(self.ticker))

        else:
            print('building model')
            self.trained = False
            dropout = 0.2
            window_size = self.n_lookback
            model = Sequential()
            model.add(LSTM(units=window_size, return_sequences=True, input_shape=(self.n_lookback, 1)))
            model.add(Dropout(rate=dropout))
            model.add(Bidirectional(LSTM(units=window_size, return_sequences=False)))
            model.add(Dense(self.n_forecast))
            model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def make_sequences(self):
        X = []
        Y = []

        for i in range(self.n_lookback, len(self.y) - self.n_forecast):
            X.append(self.y[i - self.n_lookback: i])
            Y.append(self.y[i: i + self.n_forecast])

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def preprocess_lstm(self, future=False):
        index_train = self.data.last_valid_index() + pd.Timedelta(-self.n_forecast, 'd')
        index_test = self.data.last_valid_index() + pd.Timedelta(-(self.n_forecast-1), 'd')
        self.df_train = self.data[:index_train]
        self.df_test = self.data[index_test:]
        if not future:
            y = self.df_train['close']
            y_test = self.df_test['close']
            self.y_test = y_test.values.reshape(-1,1)
            self.y = y.values.reshape(-1, 1)
        y = self.data['close']
        self.y= y.values.reshape(-1,1)
        self.y_test = self.y[-self.n_forecast:]
        # scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = self.scaler.fit(self.y)
        self.y = self.scaler.transform(self.y)
        self.y_test = self.scaler.transform(self.y_test)
        self.X, self.Y = self.make_sequences()
        self.X_train = self.make_train_X()


    def make_train_X(self):
        X = []
        for i in range(self.n_lookback, len(self.y)):
            X.append(self.y[i - self.n_lookback: i])
        X = np.array(X)
        return X

    def train(self):
        csv_logger = CSVLogger('trainings/training{}.log'.format(self.ticker), separator=',', append=False)
        self.history = self.model.fit(self.X, self.Y, epochs=25, batch_size=32, validation_split=0.2,
                                      verbose=0, shuffle=True, callbacks=[csv_logger, TqdmCallback(verbose=0)])

    def forecast(self):
        # generate the forecasts
        X_ = self.y[- self.n_lookback:]  # last available input sequence
        self.X_ = X_.reshape(1, self.n_lookback, 1)
        # evaluate the model
        score = self.model.evaluate(self.X_, self.y_test.reshape(1, self.n_forecast, 1))
        print('evaluation score: ', score)
        # predict new values
        Y_ = self.model.predict(self.X_).reshape(-1, 1)
        self.Y_ = self.scaler.inverse_transform(Y_)
        x_hat = self.model.predict(self.X_train)
        self.x_hat = self.scaler.inverse_transform(x_hat)[:,-1]
        self.estimated = np.append(self.x_hat, self.Y_[:, -1])
        range_all= pd.date_range(self.df_train.first_valid_index()+pd.Timedelta(self.n_lookback,'d'),
                              self.df_test.last_valid_index()+pd.Timedelta(self.n_forecast,'d'))
        print(self.x_hat.shape, self.Y_.shape, range_all.shape)
        self.estimated = pd.DataFrame(data= self.estimated, index=range_all, columns=['close'])
        return self.estimated




    def bool_existing(self):
        df_models = pd.read_csv('models/models.csv', names=['ticker', 'location','date'])
        if not df_models['ticker'].eq(self.ticker).any():
            return False
        else:
            last_date = df_models.date[df_models['ticker'] == self.ticker].iloc[0]
            last_date = datetime.datetime.strptime(last_date,'%Y-%m-%d').date()
            today = date.today()
            if (today - last_date).days > 20:
                return False
        return True

    def save_model(self):
        path = 'models/{}'.format(self.ticker)
        self.model.save(filepath=path)

        if not self.bool_existing():
            with open('models/models.csv', 'a') as models_list:
                dict_obj = csv.DictWriter(models_list, fieldnames=['ticker_name', 'model_location','date'])
                today = date.today()
                dict_obj.writerow({'ticker_name': self.ticker, 'model_location': path, 'date':today})
            models_list.close()

    def load_model(self):
        df_models = pd.read_csv('models/models.csv', names=['ticker', 'location','date'])
        loc = df_models['location'][df_models['ticker'] == self.ticker].values[0]
        return keras.models.load_model(loc)

    def plot_training(self):
        if self.trained:
            history = self.history
        else:
            history = self.history.history
        f0 = go.Figure(
            data=[
                go.Scatter(y=history['loss'], name="training"),
                go.Scatter(y=history['val_loss'], name="validation"),
            ],
            layout={"xaxis": {"title": "Epoch"}, "yaxis": {"title": "MSE"}, "title": "model evaluation"}
        )
        st.plotly_chart(f0, use_container_width=False, sharing="streamlit")
        print('mean mse training', np.mean(history['loss']))
        print('mean mse validation', np.mean(history['val_loss']))

    def plots(self, zoomed=False):
        f1 = go.Figure(
            data=[
                go.Scatter(x=self.data.index, y=self.data.close, name="historical", opacity=0.6,
                           marker={'color': 'blue'}),
                go.Scatter(x=self.estimated.index, y=self.estimated.close, name='predicted', marker={'color': 'pink'})
            ],
            layout={"xaxis": {"title": "Date"}, "yaxis": {"title": "price in $"},
                    "title": "Estimated prices for {}".format(self.ticker)}

        )
        st.plotly_chart(f1, use_container_width=False, sharing="streamlit")

        if zoomed:
            f2 = go.Figure(
               data=[
                   go.Scatter(x=self.data.index[-31:], y=self.data.close.iloc[-31:], name="historical", opacity=0.6,
                              marker={'color': 'blue'}),
                   go.Scatter(x=self.estimated.index[-31:], y=self.estimated.close.iloc[-31:], name='predicted',
                              marker={'color': 'pink'})
               ],)
            f2.add_vline(x=self.df_test.first_valid_index(), line_dash="dash", line_color="green")
            st.plotly_chart(f2, use_container_width=False, sharing="streamlit")

