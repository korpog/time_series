import pandas as pd
import streamlit as st
import yfinance as yf

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA as arima_model

st.write("""
# Time Series ARIMA Model
""")

ticker = 'NVDA'
ticker_data = yf.Ticker(ticker)
ticker_df = ticker_data.history(period='max', start='2019-5-31', end='2024-5-31')

st.write('NVIDIA stock price')
st.line_chart(ticker_df.Close)

