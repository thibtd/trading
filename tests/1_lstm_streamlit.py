import sys

# tell interpreter where to look
sys.path.insert(0, "..")
import streamlit as st
from modules.LSTM import lstm
from modules import stocks_processing as stocks
import pandas as pd

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

ticker = st.session_state.tick
data = stocks.get_data(ticker=ticker, period='6y')
st.session_state.data = data
data = stocks.preprocess(data)

print(data)
model = lstm(ticker, data, 60, 10)
st.markdown('Here is a summary of the architecture of the model')
st.write(model.model.summary())
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

