from builtins import print

from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
from scipy.optimize import curve_fit
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# you can get apiKey from AlphaVantage.co
from services.KEY import getApiKey
from services.KNNAlgorithm import KNNAlgo as knna

from datetime import datetime

# from scikit learn SKLearn
from sklearn import neighbors

import matplotlib
import matplotlib.pyplot as plt

class API_Stock:
    data_actual= []
    meta_data = []
    data_predictions=[]

    def readDataStock(self, symbolStock):
        ts = TimeSeries(key=getApiKey(), output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol=symbolStock, outputsize="full")
        return data, meta_data


    def getPredictData_and_actualData(self):
        predict_close=[]

        knna.predictFor()


    def knn_stock_prediction(self, symbolStock):
        data, meta_data = self.getByStockSy(symbolStock)

        # actual data
        # sumber: https://plot.ly/python/plot-data-from-csv/
        df_external_source = FF.create_table(data.head())
        py.iplot(df_external_source, filename='df-external-source-table')


        # X Itu datanya
        # y targetnya
        # ploting lebih bagus per tahun atau per bulan, jangan per hari
        knn = KNeighborsRegressor(n_neighbors=4,weights='distance', p=2, metric='euclidian')
        y = knn.fit()
            
        trace_actual = go.Scatter(
            x=data['timestamp'],
            y=data['4. close'],
            name= 'Actual Price',
            line=dict(
                color='blue',
                width = 4,
                dash = 'dash'
            )
        )


        # Hasil Prediksi
        trace_prediction = go.Scatter(

        )

        data_all = [trace_actual, trace_prediction]

        # layout
        layout = go.Layout(
            title="Stock Price Prediction Result " + symbolStock,
            plot_bgcolor='white',
            showlegend=True
        )
        fig = go.Figure(data=[trace_actual],)

        return data_graph, data_table

