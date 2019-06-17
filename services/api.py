import math

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries

from services.KEY import getApiKey
from services.KNNAlgorithm import KnnRegression

# from services.KNR import KNN_Regression

plotly.tools.set_credentials_file(username='junhin04', api_key='c0YKlYKB4vqlUGfaEDNO')
ts = TimeSeries(key=getApiKey(), output_format='pandas')


class API_Stock:

    def __init__(self, symbolstock, k):
        self.__symbolstock = symbolstock
        self.__k = k

    def getResult(self):
        stock_symbol = self.__symbolstock
        k = self.__k
        stock, meta_data = ts.get_daily_adjusted(stock_symbol, outputsize='full')
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

        test_cutoff = math.floor(len(df) / 1.5)

        test = df.loc[df.index[test_cutoff:]]
        train = df.loc[df.index[1:test_cutoff]]

        # reset index untuk lebih mudah dimapping ke dalam dictionary
        train_reset_index = train.reset_index(drop=True)
        test_reset_index = test.reset_index(drop=True)

        x_column = ['close']

        y_column = ['close_next']

        knr = KnnRegression(k)

        knr.fit(train_reset_index[x_column], train_reset_index[y_column])
        predictions = knr.predict_by_myself_method(train_reset_index[x_column], test_reset_index[x_column])

        # knr = KNeighborsRegressor(n_neighbors=6, weights='distance',p=2 )
        # knr.fit(train[x_column], train[y_column])
        # predictions = knr.predict(test[x_column])

        # convert to dataframe dan indexnya diubah supaya mengikuti index dari index test (untuk index tanggal)
        predictions = pd.DataFrame(np.row_stack(predictions), index=test.index.get_level_values(0).values)

        actual = test[y_column]

        from sklearn.metrics import mean_squared_error, r2_score

        print(math.sqrt(mean_squared_error(actual['close_next'], predictions[0])))

        r_square = r2_score(actual['close_next'], predictions[0])
        print(r_square)

        trace_actual = go.Scatter(
            x=df['date'][test_cutoff:],
            y=actual['close_next'],
            name=stock_symbol + " Actual",
            line=dict(color='red'),
            opacity=0.8

        )

        trace_predict = go.Scatter(
            x=df['date'][test_cutoff:],
            y=predictions[0],
            name=stock_symbol + " Predictions",
            line=dict(color='green'),
            opacity=0.8
        )

        data_trace = [trace_actual, trace_predict]

        layout = dict(
            title=stock_symbol + " ranges slider (" + str(r_square) + ")",
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

        # graphDiv = py.plot(fig, filename= stock_symbol + " stock price prediction" )
        # graphJSON = json.dumps(data_trace, cls=plotly.utils.PlotlyJSONEncoder)

        # linkgraph = tls.get_embed(graphDiv)
        return fig

    # def create_plot(self, dataframe, actual, prediction):
    #
    #     graphJSON = json.dumps()
