import numpy as np
import csv
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import CSVLogger
import streamlit as st
import plotly.graph_objects as go
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)


class lstm:
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
            # model.add(Bidirectional(LSTM(units=window_size,return_sequences=True )))
            # model.add(Dropout(rate=dropout))
            model.add(Bidirectional(LSTM(units=window_size, return_sequences=False)))
            model.add(Dense(self.n_forecast))
            model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def make_sequences(self):
        X = []
        Y = []

        for i in range(self.n_lookback, len(self.y) - self.n_forecast + 1):
            X.append(self.y[i - self.n_lookback: i])
            Y.append(self.y[i: i + self.n_forecast])

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def preprocess(self):
        index = self.data.last_valid_index() + pd.Timedelta(-self.n_forecast, 'd')
        self.df_train = self.data[:index]
        self.df_test = self.data[index:]
        y = self.data['close']
        y = y.values.reshape(-1, 1)
        # scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = self.scaler.fit(y)
        self.y = self.scaler.transform(y)
        self.X, self.Y = self.make_sequences()

    def train(self):
        csv_logger = CSVLogger('trainings/training{}.log'.format(self.ticker), separator=',', append=False)
        self.history = self.model.fit(self.X, self.Y, epochs=25, batch_size=32, validation_split=0.2,
                                      verbose=0, shuffle=True, callbacks=[csv_logger, TqdmCallback(verbose=0)])

    def forecast(self, training=True):
        # generate the forecasts
        X_ = self.y[- (self.n_lookback + self.n_forecast):-self.n_forecast]  # last available input sequence
        self.X_ = X_.reshape(1, self.n_lookback, 1)
        # evaluate the model
        score = self.model.evaluate(self.X_, self.y[-self.n_forecast:].reshape(1, self.n_forecast, 1))
        print('evaluation score: ', score)
        # predict new values
        Y_ = self.model.predict(self.X_).reshape(-1, 1)
        self.Y_ = self.scaler.inverse_transform(Y_)
        if training == True:
            x_hat = self.model.predict(self.X)
            x_hat = self.scaler.inverse_transform(x_hat)
            self.x_hat = np.median(x_hat, axis=1)

    def bool_existing(self):
        str_arr_csv = []
        with open('models.csv', 'rt') as c:
            str_arr_csv = c.readlines()
            c.close()
        if not self.ticker in str(str_arr_csv):
            return False
        return True

    def save_model(self):
        path = 'models/{}'.format(self.ticker)
        self.model.save(filepath=path)

        if not self.bool_existing():
            with open('models.csv', 'a') as models_list:
                dict_obj = csv.DictWriter(models_list, fieldnames=['ticker_name', 'model_location'])
                dict_obj.writerow({'ticker_name': self.ticker, 'model_location': path})
            models_list.close()

    def load_model(self):
        df_models = pd.read_csv('models.csv', names=['ticker', 'location'])
        loc = df_models['location'][df_models['ticker'] == self.ticker].values[0]
        return keras.models.load_model(loc)

    def plot_training(self):
        if self.trained == True:
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
        Y = self.scaler.inverse_transform(self.Y[:, 0, :])
        test = self.df_test.close.values
        range = pd.date_range(self.df_train.first_valid_index() + pd.Timedelta(self.n_lookback, 'd'),
                              self.df_train.last_valid_index())
        f1 = go.Figure(
            data=[
                go.Scatter(x=range, y=self.x_hat, name="estimated"),
                go.Scatter(x=range, y=Y[:, 0], name="historical", opacity=0.6),
                go.Scatter(x=pd.date_range(self.df_test.first_valid_index(), self.df_test.last_valid_index()),
                           y=self.Y_[:, 0], name='predictions'),
                go.Scatter(x=pd.date_range(self.df_test.first_valid_index(), self.df_test.last_valid_index()), y=test,
                           name='actual'),
            ],
            layout={"xaxis": {"title": "Date"}, "yaxis": {"title": "price in $"},
                    "title": "Estimated prices for {}".format(self.ticker)}
        )
        st.plotly_chart(f1, use_container_width=False, sharing="streamlit")

        if zoomed:
            range = pd.date_range(self.df_train.last_valid_index() + pd.Timedelta(-30, 'd'),
                                  self.df_train.last_valid_index())
            print('values')
            print(Y[-30:, 0])
            f2 = go.Figure(
                data=[go.Scatter(x=range, y=self.x_hat[-31:], name='estimated'),
                      go.Scatter(x=range, y=self.df_train.close.values[-31:, ], name='historical'),
                      go.Scatter(x=pd.date_range(self.df_test.first_valid_index(), self.df_test.last_valid_index()),
                                 y=self.Y_[:, 0], name='predictions'),
                      go.Scatter(x=pd.date_range(self.df_test.first_valid_index(), self.df_test.last_valid_index()),
                                 y=test, name='actual'),
                      ],
                layout={"xaxis": {"title": "Date"}, "yaxis": {"title": "price in $"},
                        "title": "Estimated prices for {} zoomed".format(self.ticker)}
            )
            f2.add_vline(x=self.df_test.first_valid_index(), line_dash="dash", line_color="green")
            st.plotly_chart(f2, use_container_width=False, sharing="streamlit")
