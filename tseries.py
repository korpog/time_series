import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pmdarima as pm

from pmdarima.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA as arima_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def print_adf_test(results):
    output = pd.Series(results[0:4], index=[
                       'Test Statistics', 'p-value', 'No. of lags', 'Number of observations'], name="ADF Test Results")
    for key, values in results[4].items():
        output[f'critical value {key}'] = values
    st.write(output)


st.set_page_config(page_title='ARIMA Time Series', layout='wide')
st.title("Time Series ARIMA Model")

with st.sidebar:

    st.header('Choose date range for the stock data')

    min_date = st.date_input(label='Enter :blue[start] date', value=datetime.date(2019, 6, 1),
                             min_value=datetime.date(2010, 1, 1),
                             max_value=datetime.date(2024, 5, 1), format="YYYY-MM-DD")

    max_date = st.date_input(label='Enter :blue[end] date', value=datetime.date(2024, 6, 1),
                             min_value=datetime.date(2010, 1, 2),
                             max_value=datetime.date(2024, 6, 1), format="YYYY-MM-DD")

    if min_date > max_date:
        st.warning('Minimum date should be earlier than maximum date')


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Data', 'ADF Test', 'ACF/PACF', 'Decomposition', 'ARIMA'])

with tab1:

    option = st.selectbox(
        "Choose stock ticker",
        ("MSFT", "AAPL", "GOOG")
    )

    ticker_data = yf.Ticker(option)
    df = ticker_data.history(
        period='max', start=min_date, end=max_date)

    st.write("Dataframe: ")
    st.dataframe(df.head())

    st.write(f'{option} stock price (Close)')
    st.line_chart(df.Close)


with tab2:
    st.header("Before differencing", divider='blue')
    res = adfuller(df.Close)
    print_adf_test(res)

    st.header("After differencing", divider='blue')
    diff = df.Close.diff().dropna()
    res2 = adfuller(diff)
    print_adf_test(res2)

with tab3:
    st.header("Before differencing", divider='blue')
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    ax1.plot(df.Close)
    fig = plot_acf(df.Close, ax=ax2)
    st.pyplot(fig)

    st.header("After differencing", divider='blue')
    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(16, 4))
    ax3.plot(diff)
    plot_acf(diff, ax=ax4)
    plot_pacf(diff, ax=ax5)
    st.pyplot(fig2)

with tab4:
    st.subheader('Time Series Decomposition')

    decomp_model = st.selectbox(
        "Choose additive or multiplicative decomposition",
        ("additive", "multiplicative",)
    )

    period = st.slider("Set period for the seasonal decomposition",
                       min_value=1, max_value=365, value=30)

    decomp = seasonal_decompose(x=df.Close, model=decomp_model, period=period)

    T, S, R = decomp.trend, decomp.seasonal, decomp.resid

    st.subheader('Trend')
    st.line_chart(T)
    st.subheader('Seasonality')
    st.line_chart(S)
    st.subheader('Residual')
    st.line_chart(R)

    fig, ax = plt.subplots()
    ax.plot(df.Close, c='blue', label='Actual value')
    ax.plot(T + S, c='red', label='Trend + Seasonal')
    ax.legend()
    st.pyplot(fig)

with tab5:
    y = df.Close
    train, test = train_test_split(y, train_size=0.95)

    model = pm.auto_arima(train)
    forecast = model.predict(test.shape[0])

    x = df.index
    fig, ax = plt.subplots()
    ax.plot(x[:len(train)], train, c='blue', label='Actual value')
    ax.plot(x[len(train):], forecast, c='green', label='Prediction')
    ax.legend()
    st.pyplot(fig)
