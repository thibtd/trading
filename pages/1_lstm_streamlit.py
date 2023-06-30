import sys

# tell interpreter where to look
sys.path.insert(0, "..")
import streamlit as st
from modules.LSTM import lstm
import stocks
import pandas as pd

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

ticker = st.session_state.tick
data = stocks.get_data(ticker=ticker, period = '6y')
print(data)
model = lstm(ticker, data, 60, 10)
st.markdown('Here is a summary of the architecture of the model')
model.model.summary()
model.preprocess()
if not model.trained:
    st.markdown('The model is being trained on your data ')
    model.train()
else:
    st.markdown('The model is already trained')
model.save_model()
model.forecast()
model.plot_training()
model.plots(zoomed=True)
