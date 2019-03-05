from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
from scipy.optimize import curve_fit
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# you can get apiKey from AlphaVantage.co
from services.KEY import getApiKey
from services.KNNAlgorithm import KNNAlgo as knna

from datetime import datetime, date

# from scikit learn SKLearn
from sklearn import neighbors

import matplotlib
import matplotlib.pyplot as plt

# https://github.com/vishwajeetv/stock_prediction/blob/master/backend/predictorWeekly.py
def normalize(df):
    normalizedValues = (df- df.mean())/df.std()
    return normalizedValues

def readData():
    symbol = "ADHI.JKT"
    ts = TimeSeries(key=getApiKey(), output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    # print(data[data.columns[0:7]])
    # print(meta_data)
    data_sorted_by_date = data.sort_values('date')
    # data_sorted_by_date['date'] = pd.to_datetime(data_sorted_by_date['date'])

    sorted_low_price = data_sorted_by_date['3. low'].values
    sorted_high_price = data_sorted_by_date['2. high'].values
    sorted_open_price = data_sorted_by_date['1. open'].values
    sorted_close_price = data_sorted_by_date['5. adjusted close'].values

    # dari ko Julio
    low_price = data['3. low'].values
    high_price = data['2. high'].values
    open_price = data['1. open'].values
    close_price = data['5. adjusted close'].values

    sorted_date = data_sorted_by_date.index.get_level_values(level='date')

    # date = data_sorted_by_date.index.get_level_values(level='date')

    # numpy, X
    data_numpy_form = np.stack(
        (sorted_date, sorted_open_price, sorted_low_price, sorted_high_price, sorted_close_price), axis=1)

    # pandas df, pd, dll
    df = pd.DataFrame(data_numpy_form, columns=['date', 'open', 'low', 'high', 'close'])

def main():
    plotly.tools.set_credentials_file(username='junhin04', api_key='c0YKlYKB4vqlUGfaEDNO')
    symbol = "ADHI.JKT"
    ts = TimeSeries (key=getApiKey(), output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    # print(data[data.columns[0:7]])
    # print(meta_data)
    data_sorted_by_date = data.sort_values('date')
    # data_sorted_by_date['date'] = pd.to_datetime(data_sorted_by_date['date'])

    sorted_low_price = data_sorted_by_date['3. low'].values
    sorted_high_price = data_sorted_by_date['2. high'].values
    sorted_open_price = data_sorted_by_date['1. open'].values
    sorted_close_price = data_sorted_by_date['5. adjusted close'].values


    # dari ko Julio
    low_price = data['3. low'].values
    high_price = data['2. high'].values
    open_price = data['1. open'].values
    close_price = data['5. adjusted close'].values

    sorted_date = data_sorted_by_date.index.get_level_values(level='date')

    # date = data_sorted_by_date.index.get_level_values(level='date')

    # numpy, X
    data_numpy_form = np.stack((sorted_date, sorted_open_price, sorted_low_price, sorted_high_price, sorted_close_price), axis=1)

    # pandas df, pd, dll
    df = pd.DataFrame(data_numpy_form, columns=['date','open','low', 'high', 'close'])

    # untuk bedain data 2016, 2017, 2018
    data_2016_actual = df[(df['date'] >= '2016-01-01') & (df['date'] <= '2016-12-31')]
    data_2017_actual = df[(df['date'] >= '2017-01-01') & (df['date'] <= '2017-12-31')]
    data_2018_actual = df[(df['date'] >= '2018-01-01') & (df['date'] <= '2018-12-31')]

    # sumber
    # https://github.com/nileshbhadana/ML/blob/835af9dc58e4341e17507effa9ecfe85bbdc03be/stock_prediction.py

    price_close_pd = close_price.reshape(-1,1)

    day = []
    for i in range(len(data_2016_actual)):
        day.append(i)

    day_count = np.array(day).reshape(-1,1)


    X_train, X_test, y_train, y_test = train_test_split(day_count, data_2016_actual['close'], train_size=0.67, shuffle=False)


    knr = KNeighborsRegressor(n_neighbors=4, weights='distance', metric='euclidean')

    knr_np= knr.fit(X_train, y_train)
    knr_np_predict = knr_np.predict(X_test)
    print(knr.score(X_train, y_train))

    trace_actual = go.Scatter(
        x = data_2016_actual['date'],
        y= data_2016_actual['close'],
        name = "ADHI ACTUAL",
        line=dict(color='#17BECF'),
        opacity= 0.8
    )
    trace_predict = go.Scatter(
        x =data_2016_actual['date'],
        y= np.concatenate((y_train,knr_np_predict), axis=0),
        name = "ADHI PREDICT",
        line=dict(color='#7F7F7F'),
        opacity=0.8
    )

    data_all = [trace_actual, trace_predict]

    # data_all = [trace_actual]
    layout = dict(
        title = 'ADHI ranges slider',
        xaxis = dict(
            rangeselector = dict(
                buttons = list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        )
    )

    fig = dict(data = data_all, layout=layout)
    py.plot(fig, filename = "ADHI stock price prediction")
    # print(data_sorted)

    # plt.plot(X_pd_date, knr_np_predict, 'k')

if __name__ == '__main__':
    main()
