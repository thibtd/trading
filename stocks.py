import numpy as np
import yfinance as yf
import pandas as pd
from stockstats import wrap, unwrap
import seaborn as sns
import datetime
import streamlit as st
import plotly.express as px


def get_data(ticker='BABA', starting=None, ending=None, short_t=5, long_t=10, period=None):
    ticker = yf.Ticker(ticker)
    if period is not None:
        time_data = ticker.history(period=period)
    else:
        time_data = ticker.history(start=starting, end=ending)
    data = wrap(time_data)
    data['rsi'] = data['rsi']
    short_sma = 'close_{}_sma'.format(short_t)
    long_sma = 'close_{}_sma'.format(long_t)
    data[short_sma] = data[short_sma]
    data[long_sma] = data[long_sma]
    data.rename(columns={'close_{}_sma'.format(short_t): 'short_sma', 'close_{}_sma'.format(long_t): 'long_sma'},
                inplace=True)
    data.drop(['dividends', 'stock splits'], axis=1, inplace=True)
    data = data.asfreq('d')
    data.fillna(method='ffill', inplace=True)
    if 'rs_14' in data.columns:
        data.drop('rs_14', axis=1, inplace=True)
    return data


def preprocess(df):
    df = df.copy()
    print(df.columns)
    df['rsi_signal'] = df.apply(lambda row: rsi_sell_bool(row['rsi']), axis=1)
    df['short_sma_prev'] = df['short_sma'].shift(1, fill_value=0)
    df['long_sma_prev'] = df['long_sma'].shift(1, fill_value=0)
    df['sma_signal'] = df.apply(lambda row: sma_cross(row['short_sma'], row['short_sma_prev'], row["long_sma"],
                                                      row['long_sma_prev']), axis=1)
    return df


# logic for rsi
# return true for a sell signal, false for a buy signal and None if nothing is happening.
def rsi_sell_bool(row):
    if row >= 70:
        # print('sell rsi', np.round(row, 2))
        # st.write('sell rsi', np.round(df['rsi'].iloc[-t], 2))
        return True
    elif row <= 30:
        # print('buy rsi', np.round(row, 2))
        # st.write('buy rsi', np.round(df['rsi'].iloc[-t], 2))
        return False


# logic for sma
# returns true is sell signal, false for buy signal, None for no signal
def sma_cross(short, short_prev, long, long_prev):
    if short_prev > long_prev and short <= long:
        return True
    elif short_prev < long_prev and short >= long:
        return False


def sma_sell_cross(df):
    if df['close_5_sma'].iloc[-2] > df['close_10_sma'].iloc[-2] and \
            df['close_5_sma'].iloc[-1] <= df['close_10_sma'].iloc[-1]:
        print('sell sma')
        # st.write('sell sma')
        return True
    elif df['close_5_sma'].iloc[-2] < df['close_10_sma'].iloc[-2] and \
            df['close_5_sma'].iloc[-1] >= df['close_10_sma'].iloc[-1]:
        print('buy sma')
        # st.write('buy sma')
        return False


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


if __name__ == "__main__":
    # get data
    symbols = pd.read_csv('symbols.csv')
    option = symbols['name']
    nvda = symbols.index[symbols['symbol'] == 'NVDA'].to_list()[0]
    st.set_page_config(
        page_title="Home",
        page_icon="💰",
    )
    st.title('stock analysis')
    st.header('Welcome to this trading helper ')
    col1, col2 = st.columns(2)

    with col1:
        stock = st.selectbox(
            'Enter the symbol for the desired stock',
            options=option, index=nvda)
        # tick = st.text_input('Enter the symbol for the desired stock', 'NVDA')
        start = st.date_input('Start date', value=datetime.date(2022, 9, 30), max_value=datetime.date.today())
        end = st.date_input('End date', value=datetime.date.today(),
                            max_value=datetime.date.today())
    with col2:
        short_sma = st.number_input('Enter the value for the short sma',
                                    value=5)
        long_sma = st.number_input('Enter the value for the long sma',
                                   value=10)
        time = st.number_input('Enter the maximum days between RSI and SMA crossing ',
                               value=9)
        delay = st.number_input('allow delay',
                                value=2)

    print(stock)
    st.session_state.tick = symbols.query("name=='{}'".format(stock))['symbol'].iloc[0]

    print(st.session_state.tick)
    df = get_data(st.session_state.tick, starting=start, ending=end, short_t=short_sma, long_t=long_sma)
    # TODO: ADD preprocessing step for the signals
    df = preprocess(df)
    st.subheader('Data for the last {} days for {} ({})'.format(time, stock, st.session_state.tick))
    st.write(df.tail(time))
    signal = get_signal(df, time, delay)
    sell = 'hold or wait'
    if signal:
        sell = 'Both indicators show that it is time to sell'
    elif signal is False:
        sell = 'Both indicators show that it is time to buy'
    st.subheader('Based on the rsi and the SMA crossing indicator:')
    st.markdown('**:blue[{}]** !'.format(sell))
    st.markdown('Have a look at the following 2 graphs for more detailed information.')
    st.markdown('The first figure depicts the closing price of {} over the time span that was selected as well as '
                'both SMA curves: '.format(st.session_state.tick))
    st.markdown('   - One using a 5 days average, can be seen as a short period. ')
    st.markdown('   - The second one using a 10 days average, can be seen as long period.')
    st.markdown('This is the first indicator that is used, when the two sma curves cross, it can be interpreted as '
                'the fact that '
                'the closing price as reached either a minima or a maxima. Hence, hinting at a change of direction of '
                'the stock. '
                'And thus can be used as a buying or selling indicator. ')
    st.markdown('The second graph depicts the RSI vale over the selected time span. The two dashed lines that can be'
                'seen are the two thresholds used for this indicator. The top one being 70 and the bottom one 30.')
    st.markdown('This is the second indicator used. The crossing of one of the thresholds by the RSI value can be seen '
                'as either a buying or selling signal depending on which threshold is crossed.')
    st.markdown('Lastly, The two indicators are used in combination in order to determine what action to take with the '
                'selected stock. As it usually happens later, the SMA as used as a starting point for a signal. Once '
                'it sends a signal, the RSI over the last {} days (the number you chose above) is checked to see if it'
                'gives the same signal.'.format(time))

    fig = px.line(df, y=['close', 'short_sma', 'long_sma'], title='Price and SMA of {}'.format(st.session_state.tick))
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
    fig2 = px.line(df, y=['rsi'], title='RSI, 14 days')
    fig2.add_hline(y=70, line_width=3, line_dash="dash", line_color="green")
    fig2.add_hline(y=30, line_width=3, line_dash="dash", line_color="green")
    st.plotly_chart(fig2, use_container_width=False, sharing="streamlit")
