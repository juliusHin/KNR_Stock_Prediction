from math import sqrt

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# from services.KNR import KNN_Regression

plotly.tools.set_credentials_file(username='junhin04', api_key='c0YKlYKB4vqlUGfaEDNO')
ts = TimeSeries(key='OGL1H2UML6HMY9CL', output_format='pandas')


class API_Stock:

    def __init__(self, symbolstock):
        self.__symbolstock = symbolstock
        # self.__k = k

    def getResult(self):
        stock_symbol = self.__symbolstock
        # k = self.__k
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
        date = df['date']

        distance_column_1 = ['close']
        distance_column_2 = ['date', 'close']
        distance_column_3 = ['close', 'daily_return']
        distance_column_4 = ['date', 'close', 'daily_return']

        k_size = range(1, int(sqrt(len(df))))
        neighbors = filter(lambda x: x % 2 != 0, k_size)
        k_scores = []

        x_column = ['close']

        y_column = ['close_next']

        # untuk ambil nilai dalam kolom tertentu
        X = df['close'].values
        y = df['close_next'].values

        # untuk jadikan X dan y satu kolom, banyak baris
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)

        scaler = StandardScaler(with_mean=True, with_std=True)
        # scaler = scaler.fit(X)
        normalized_X = scaler.fit_transform(X)

        scaler_label = StandardScaler(with_mean=True, with_std=True)
        # scaler_label = scaler_label.fit(y)
        normalized_y = scaler_label.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(normalized_X, normalized_y, train_size=0.8, shuffle=False)

        knr = KNeighborsRegressor(weights='distance', p=2, metric='euclidean')
        param_grid = {'n_neighbors': np.arange(1, int(sqrt(len(x_train))), 2)}
        knr_gscv = GridSearchCV(knr, param_grid, cv=5)
        knr_gscv.fit(x_train, y_train)

        # untuk dapat parameter neighbors yang terbaik
        best_n_value = knr_gscv.best_params_

        knr_fix = KNeighborsRegressor(n_neighbors=best_n_value['n_neighbors'], weights='distance', metric='euclidean')
        knr_fix.fit(x_train, y_train)

        predictions = knr_fix.predict(x_test)

        rmse = sqrt(mean_squared_error(y_test, predictions))

        # hasil prediksi yang di-inverse_transform untuk mengembalikan nilai yang sudah distandarisasi menjadi normal
        # atau bahasa sederhananya dilakukan DENORMALISASI
        predictions = scaler.inverse_transform(predictions)
        predictions = pd.DataFrame(data=predictions, index=range(len(x_train), len(x_train) + len(x_test))).round(
            decimals=3)

        y_test = scaler_label.inverse_transform(y_test)

        actual = y_test
        actual = pd.DataFrame(data=actual, index=range(len(y_train), len(y_train) + len(y_test))).round(decimals=3)

        r_square = r2_score(actual, predictions)
        r_square = r_square * 100.0

        table_trace = go.Table(
            header=dict(
                values=['Date', 'Predictions', 'Actual'],
                fill=dict(color='#C2D4FF'),
                align=['left'] * 5
            ),
            cells=dict(
                values=[
                    df.date,
                    predictions,
                    actual
                ],
                align=['left'] * 5
            )
        )

        cut_off = int(len(x_test) + 1)
        trace_actual = go.Scatter(
            x=df['date'][cut_off:],
            y=actual[0],
            name=stock_symbol + " Actual",
            line=dict(color='red'),
            opacity=0.8

        )

        trace_predict = go.Scatter(
            x=df['date'][cut_off:],
            y=predictions[0],
            name=stock_symbol + " Predictions",
            line=dict(color='green'),
            opacity=0.8
        )

        data_trace = [trace_actual, trace_predict]

        layout = dict(
            title=stock_symbol + "\n (RMSE: " + str(rmse) + ")\n" + "Accuracy (" + str(r_square) + "%)",
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
            ),
            height=800
        )

        fig = dict(data=data_trace, layout=layout)
        date_df = pd.DataFrame(data=df['date'], index=range(len(x_train), len(x_train) + len(x_test)))
        # index = range(len(x_train), len(x_train)+len(x_test))
        table_df = pd.concat([date_df, predictions, actual], axis=1)
        table_df.columns = ['date', 'predictions', 'actual']

        return fig, table_df
