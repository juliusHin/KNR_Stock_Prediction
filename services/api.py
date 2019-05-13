import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from alpha_vantage.timeseries import TimeSeries
from services.KEY import getApiKey
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from services.KNR import KNN_Regression

class API_Stock:

    plotly.tools.set_credentials_file(username='junhin04', api_key='c0YKlYKB4vqlUGfaEDNO')
    ts = TimeSeries(key=getApiKey(), output_format='pandas')

    def __init__(self, symbolstock, k):
        self.__symbolstock = symbolstock
        self.__k = k

    '''
    sumber
    https://github.com/DavidCico/Self-implementation-of-KNN-algorithm/blob/master/Abalone_case_study_regression.ipynb
    '''
    ## Convert string column to float
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    ## Convert string column to integer
    def str_column_to_int(self,dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    def knn_stock_prediction(self):
        stock, meta_data = self.ts.get_daily_adjusted(self.__symbolstock,outputsize='full')
        stock = stock.sort_values('date')

        open_price = stock['1. open'].values
        low = stock['3. low'].values
        high = stock['2. high'].values
        close = stock['4. close'].values
        sorted_date = stock.index.get_level_values(level='date')

        stock_numpy_format = np.stack((sorted_date, open_price, low
                                       , high, close), axis=1)
        df = pd.DataFrame(stock_numpy_format, columns=['date', 'open', 'low', 'high', 'close'])

        df = df[df['open'] > 0]
        df = df[(df['date'] >= "2016-01-01") & (df['date'] <= "2018-12-31")]
        df = df.reset_index(drop=True)

        df['close_next'] = df['close'].shift(-1)
        df['daily_return'] = df['close'].pct_change(1)
        df.fillna(0, inplace=True)

        distance_column_1 = ['close']
        distance_column_2 = ['date', 'close']
        distance_column_3 = ['close', 'daily_return']
        distance_column_4 = ['date', 'close', 'daily_return']

        test_cutoff = math.floor(len(df) / 3)
        test = df.loc[df.index[1:test_cutoff]]
        train = df.loc[df.index[test_cutoff:]]

        # x_column = ['close', 'daily_return']

        # x_column = ['date', 'close']

        for x in range (1, len(df)):
            self.str_column_to_float(df,x)

        self.str_column_to_int(df,0)

        x_column = ['close']

        y_column = ['close_next']

        # predictions = KNR.knn_regressor(train[x_column],train[y_column], test[x_column], 3)
        # a = train[y_column]
        # b = test[x_column]

        predictions = []

        knr = KNN_Regression(self.__k)
        knr.fit(train[x_column], train[y_column])
        for i in len(test[x_column]):
            neighbors = knr.getNeighbors(train[y_column], test[x_column].iloc[i],self.__k)
            output = knr.getRegression(neighbors)
            predictions.append(output)

        actual = test[y_column]

        print(math.sqrt(metrics.mean_squared_error(actual, predictions)))

        trace_actual = go.Scatter(
            x=df['date'],
            y=actual['close_next'],
            name=self.__symbolstock+ " Actual",
            line=dict(color='red'),
            opacity=0.8

        )

        trace_predict = go.Scatter(
            x=df['date'],
            y=predictions,
            name=self.__symbolstock + " Predictions",
            line=dict(color='green'),
            opacity=0.8
        )

        data_trace = [trace_actual, trace_predict]

        layout = dict(
            title=self.__symbolstock + " ranges slider",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
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

        fig = dict(data=data_trace, layout=layout)
        return py.plot(fig, filename=self.__symbolstock + " stock price prediction")

if __name__ == '__main__':
    tes = API_Stock('WIKA.JKT',3)
    tes.knn_stock_prediction()