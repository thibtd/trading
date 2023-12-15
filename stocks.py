import numpy as np
import yfinance as yf
import pandas as pd
from stockstats import wrap, unwrap
import seaborn as sns
import datetime
from modules.LSTM import lstm
from modules import stocks_processing as stocks
import streamlit as st
import plotly.express as px


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
    st.markdown('Disclaimer: This is not financial advice. This tool exists for educational purposes only.')
    st.header('First simple analysis')

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
        st.session_state.time = time
        st.session_state.delay = delay
        st.session_state.start = start

    print(stock)
    st.session_state.tick = symbols.query("name=='{}'".format(stock))['symbol'].iloc[0]

    print(st.session_state.tick)
    df = stocks.get_data(st.session_state.tick, starting=st.session_state.start, ending=end)
    st.session_state.data = df
    df = stocks.preprocess(df,short_t=short_sma, long_t=long_sma)
    st.subheader('Data for the last {} days for {} ({})'.format(time, stock, st.session_state.tick))
    st.write(df.tail(time))
    signal = stocks.get_signal(df, time, delay)
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

    st.header('Using AI to predict movements')
    st.markdown('')

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False


    def click_button():
        st.session_state.clicked = True


    st.button('Create model', on_click=click_button)

    if st.session_state.clicked:
        ticker = st.session_state.tick
        data = stocks.get_data(ticker=ticker, period='6y')
        st.session_state.data = data
        data = stocks.preprocess(data)

        print(data)
        model = lstm(ticker, data, 60, 10)
        #st.markdown('Here is a summary of the architecture of the model')
        #st.write(model.model.summary())
        model.preprocess_lstm(future=True)
        if not model.trained:
            st.markdown('The model is being trained on your data ')
            model.train()
        else:
            st.markdown('The model is already trained')
        model.save_model()
        st.session_state.forecast = model.forecast()
        model.plot_training()
        model.plots(zoomed=False)
        # Indicators using predictions
        st.header('Determining the indicators using the predicted values')
        df_shortened = st.session_state.forecast[st.session_state.forecast.index.date >= st.session_state.start]
        data_yahoo = st.session_state.data
        df_shortened['high'] = data_yahoo['High']
        df_shortened['low'] = data_yahoo['Low']
        df_shortened['volume'] = data_yahoo['Volume']
        df_shortened.index.name = 'date'
        df_shortened = stocks.preprocess(df_shortened)
        st.write(df_shortened.tail())
        signal = stocks.get_signal(df_shortened, st.session_state.time, st.session_state.delay)
        sell = 'Hold or wait'
        if signal:
            sell = 'Both indicators show that it is time to sell'
        elif signal is False:
            sell = 'Both indicators show that it is time to buy'
        st.subheader('Based on the rsi and the SMA crossing indicator:')
        st.markdown('**:blue[{}]** !'.format(sell))
        fig = px.line(df_shortened, y=['close', 'short_sma', 'long_sma'],
                      title='Price and SMA of {}'.format(st.session_state.tick))
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
        fig2 = px.line(df_shortened, y=['rsi'], title='RSI, 14 days')
        fig2.add_hline(y=70, line_width=3, line_dash="dash", line_color="green")
        fig2.add_hline(y=30, line_width=3, line_dash="dash", line_color="green")
        st.plotly_chart(fig2, use_container_width=False, sharing="streamlit")

