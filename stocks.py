import numpy as np
import yfinance as yf
import pandas as pd
from stockstats import wrap, unwrap
import seaborn as sns
import datetime
import streamlit as st
import plotly.express as px


def get_data(tick='BABA', start=None, end=None):
    ticker = yf.Ticker(tick)
    time_data = ticker.history(start=start,
                               end=end)
    data = wrap(time_data)
    data['rsi'] = data['rsi']
    data['close_5_sma'] = data['close_5_sma']
    data['close_10_sma'] = data['close_10_sma']
    if 'rs_14' in data.columns:
        data.drop('rs_14', axis=1, inplace=True)
    return data


# logic for rsi
# return true for a sell signal, false for a buy signal and None if nothing is happening.
def rsi_sell_bool(df, time=1):
    t = -time
    if time > 1:
        for i in range(time, 1, -1):
            if df['rsi'].iloc[-i] >= 70:
                print('sell rsi', np.round(df['rsi'].iloc[-i], 2))
                st.write('sell rsi', np.round(df['rsi'].iloc[-i], 2))
                return True
            elif df['rsi'].iloc[-i] <= 30:
                print('buy rsi', np.round(df['rsi'].iloc[-i], 2))
                st.write('buy rsi', np.round(df['rsi'].iloc[-i], 2))
                return False
    else:
        if df['rsi'].iloc[t] >= 70:
            print('sell rsi', np.round(df['rsi'].iloc[t], 2))
            st.sell('buy rsi', np.round(df['rsi'].iloc[t], 2))
            return True
        elif df['rsi'].iloc[t] <= 40:
            print('buy rsi', np.round(df['rsi'].iloc[t], 2))
            st.write('buy rsi', np.round(df['rsi'].iloc[t], 2))
            return False


# logic for sma
# returns true is sell signal, false for buy signal, None for no signal
def sma_sell_cross(df):
    if df['close_5_sma'].iloc[-2] > df['close_10_sma'].iloc[-2] and \
            df['close_5_sma'].iloc[-1] <= df['close_10_sma'].iloc[-1]:
        print('sell sma')
        st.write('sell sma')
        return True
    elif df['close_5_sma'].iloc[-2] < df['close_10_sma'].iloc[-2] and \
            df['close_5_sma'].iloc[-1] >= df['close_10_sma'].iloc[-1]:
        print('buy sma')
        st.write('buy sma')
        return False


# main logic
# returns true if both signals indicate sell, false if both signal indicate buy
def get_signal(time, df):
    sma = sma_sell_cross(df)
    rsi = rsi_sell_bool(df, time)
    if sma == True and rsi == True:
        print('sell sma and rsi')
        st.write('sell sma and rsi')
        return True
    elif sma == False and rsi == False:
        print('buy sma and rsi')
        st.write('buy sma and rsi')
        return False


if __name__ == "__main__":
    # get data
    st.title('stock analysis')
    st.header('Welcome to this trading helper ')
    tick = st.text_input('Enter the symbol for the desired stock', 'BABA')
    st.write('The current movie title is', tick)
    start = st.date_input('start date', value=datetime.date(2022,9,30))
    end = st.date_input('end date', value=datetime.date(2022, 11, 5), min_value=start)
    st.write(start, end)
    df = get_data(tick, start=start, end=end)

    time = 9
    st.write(df.tail(time))
    get_signal(time, df)
    fig = px.line(df, y=['close','close_5_sma','close_10_sma'], title='Price and sma of {}'.format(tick))
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")
    fig2 = px.line(df,y=['rsi'], title= 'RSI, 14 days')
    fig2.add_hline(y=70, line_width=3, line_dash="dash", line_color="green")
    fig2.add_hline(y=30, line_width=3, line_dash="dash", line_color="green")
    st.plotly_chart(fig2, use_container_width=False, sharing="streamlit")