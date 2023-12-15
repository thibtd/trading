import numpy as np
import yfinance as yf
import pandas as pd
from stockstats import wrap, unwrap
import seaborn as sns
import datetime
import streamlit as st
import plotly.express as px


@st.cache_data
def get_data(ticker='BABA', starting=None, ending=None, period=None):
    """
    Function to retrieve data from yfinance
    Parameters
    ----------
    ticker: ticker of the desired stock
    starting: starting date of the stock
    ending: ending date of the stock
    period: the periodicity of the data (days,...)

    Returns
    time_data: dataFrame with the retrieved data.
    -------

    """
    ticker = yf.Ticker(ticker)
    if period is not None:
        time_data = ticker.history(period=period)
    else:
        time_data = ticker.history(start=starting, end=ending)

    return time_data


def preprocess(df, short_t=5, long_t=10):
    """
    Function to preprocess the data. Adds information such as rsi, long SMA and short SMA

    :param df: dataset
    :param short_t: time of the short SMA
    :param long_t: time of the long SMA
    :return: preprocessed df
    """
    df = df.copy()
    print('hello')
    print(df.info())
    print(df.columns)
    print(' death')
    df = wrap(df)
    print(df)
    df['rsi'] = df['rsi']
    print(df.columns)
    short_sma = 'close_{}_sma'.format(short_t)
    long_sma = 'close_{}_sma'.format(long_t)
    df[short_sma] = df[short_sma]
    df[long_sma] = df[long_sma]
    df.rename(columns={'close_{}_sma'.format(short_t): 'short_sma', 'close_{}_sma'.format(long_t): 'long_sma'},
              inplace=True)
    df['rsi_signal'] = df.apply(lambda row: rsi_sell_bool(row['rsi']), axis=1)
    df['short_sma_prev'] = df['short_sma'].shift(1, fill_value=0)
    df['long_sma_prev'] = df['long_sma'].shift(1, fill_value=0)
    df['sma_signal'] = df.apply(lambda row: sma_cross(row['short_sma'], row['short_sma_prev'], row["long_sma"],
                                                      row['long_sma_prev']), axis=1)
    if 'dividends' in df.columns:
        df.drop('dividends', axis=1, inplace=True)
    if 'stock splits' in df.columns:
        df.drop('stock splits', axis=1, inplace=True)
    if 'rs_14' in df.columns:
        df.drop('rs_14', axis=1, inplace=True)
    df = df.asfreq('d')
    df.fillna(method='ffill', inplace=True)
    return df


# logic for rsi
# return true for a sell signal, false for a buy signal and None if nothing is happening.
def rsi_sell_bool(row):
    """
    Logic for the RSI indicator
    Parameters
    ----------
    row: the current value of the RSI.

    Returns
    True if sell signal, False if buy signal, None if no signal
    -------

    """
    if row >= 70:
        # print('sell rsi', np.round(row, 2))
        # st.write('sell rsi', np.round(df['rsi'].iloc[-t], 2))
        return True
    elif row <= 30:
        # print('buy rsi', np.round(row, 2))
        # st.write('buy rsi', np.round(df['rsi'].iloc[-t], 2))
        return False
    else:
        return None


# logic for sma
# returns true is sell signal, false for buy signal, None for no signal
def sma_cross(short, short_prev, long, long_prev):
    """
    Function to compute the logic for the sma crossing based on a long a short signal.
    Parameters
    ----------
    short: the value of the short SMA at time t.
    short_prev: the value of the short SMA at time t-1
    long: the value of the long SMA at time t.
    long_prev: the value of the long SMA at time t-1

    Returns
    True if sell signal, for for buy signal, None for no signal
    -------
    """
    if short_prev > long_prev and short <= long:
        return True
    elif short_prev < long_prev and short >= long:
        return False


# def sma_sell_cross(df):
#     if df['close_5_sma'].iloc[-2] > df['close_10_sma'].iloc[-2] and \
#             df['close_5_sma'].iloc[-1] <= df['close_10_sma'].iloc[-1]:
#         print('sell sma')
#         # st.write('sell sma')
#         return True
#     elif df['close_5_sma'].iloc[-2] < df['close_10_sma'].iloc[-2] and \
#             df['close_5_sma'].iloc[-1] >= df['close_10_sma'].iloc[-1]:
#         print('buy sma')
#         # st.write('buy sma')
#         return False


# main logic
# returns true if both signals indicate sell, false if both signal indicate buy
def get_signal(df, t, delay):
    # sma = sma_sell_cross(df)
    # rsi = rsi_sell_bool(df)
    # st.write(rsi)
    sma = None
    rsi = None
    if True in df['sma_signal'].iloc[-delay:]:
        sma = True
    elif False in df['sma_signal'].iloc[-delay:]:
        sma = False
    if True in df['rsi_signal'].iloc[-t:]:
        rsi = True
    elif False in df['rsi_signal'].iloc[-t:]:
        rsi = False

    if sma is True and rsi is True:
        print('sell sma and rsi')
        # st.write('sell sma and rsi')
        return True
    elif sma is False and rsi is False:
        print('buy sma and rsi')
        # st.write('buy sma and rsi')
        return False
