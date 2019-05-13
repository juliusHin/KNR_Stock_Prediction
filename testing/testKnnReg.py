from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
# from numpy.core.tests.test_mem_overlap import xrange
# from scipy._lib.six import xrange
from pandas import cut
from scipy.optimize import curve_fit
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# you can get apiKey from AlphaVantage.co
from services.KEY import getApiKey
from services.KNNAlgorithm import KNNAlgo as knna

from datetime import datetime, date

# from scikit learn SKLearn
from sklearn import neighbors

import matplotlib
import matplotlib.pyplot as plt

import math
import operator
from collections import Counter
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from collections import defaultdict


plotly.tools.set_credentials_file(username='junhin04', api_key='c0YKlYKB4vqlUGfaEDNO')
ts = TimeSeries(key=getApiKey(), output_format='pandas')

class KnnBase(object):

    """
    def __init__(self, k, weights=None):
        self.k = k
        self.weights = weights

    """

    def __init__(self, k):
        self.k = k
        # self.weights = weights

    def list_to_npArray(vector1, vector2):
        '''convert the list to numpy array'''
        if type(vector1) == list:
            vector1 = np.array(vector1)
        if type(vector2) == list:
            vector2 = np.array(vector2)
        return vector1, vector2


    def euclidean_distance(self, data_point1, data_point2):
        # if len(data_point1) != len(data_point2) :
        #     raise ValueError('feature length not matching')
        # else:
        #     distance = np.sqrt(sum((data_point2 - data_point1) ** 2))

        return math.sqrt(sum((data_point1 - data_point2) ** 2))

        # dist = [(a-b)**2 for a, b in zip(data_point1, data_point2)]
        # dist = math.sqrt(sum(dist))
        # return dist

        # return distance.euclidean(data_point1, data_point2)

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        return self

    def get_neighbors(self, train_set_data_points, test_feature_data_point, k):
        distances = []
        # length = len(test_feature_data_point)-1
        length = (train_set_data_points.shape[0])-1
        for index in range(len(train_set_data_points)):
            dist = self.euclidean_distance(test_feature_data_point, train_set_data_points.iloc[index],length)
            distances.append((train_set_data_points[index], dist, index))
        # distances = distance.euclidean(train_set_data_points, test_feature_data_point)
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for index in range(k):
            neighbors.append(distances[index][0])
        return neighbors

    def get_neighbors_v(self, train_set, test_set, k):
        ''' return k closet neighbour of test_set in training set'''
        # calculate euclidean distance
        euc_distance = np.sqrt(np.sum((train_set - test_set) ** 2, axis=1))
        # return the index of nearest neighbour
        return np.argsort(euc_distance)[0:k]

    def distance_weighted(self, distances):

        """
        distance Weighted

        W(x,p_i) = exp(-Distance(x,p_i)) / sum_from_i=1_to_k exp(-Distance(x,p_i))

        """
        matches = [(1, y) for d, y in distances if d == 0]
        return matches if matches else [(1/d, y) for d, y in distances]


    def y_predict(self, train_set, test_set, k):
        """
        distance Weighted Regression

        y_predictions = sum_from_i=1_to_k W (X_0, X_i)y_i

        """
        distance = self.get_neighbors_v(train_set, test_set, k)
        dist = 1. / distance
        inf_mask = np.isinf(dist)
        inf_row = np.any(inf_mask,axis=1)
        dist[inf_row] = inf_mask[inf_row]
        return dist


class KnnRegression(KnnBase):
    # def _predict_one(self, test_feature_data_point):
    #     nearest_neighbors = self.get_neighbors(self.train_X, test_feature_data_point, self.k)
    #     # weighted_distance = self.distance_weighted(distances[: self.k])
    #     # weights_by_class = defaultdict(list)
    #     total_val = 0.0
    #     for index in nearest_neighbors:
    #         total_val += self._y[index]
    #
    #     return total_val/self.k

    def predict(self, X):
        # return [self._predict_one(x) for x in X]
        y_pred = np.empty(())

    def predict_by_myself_method(self, train, test):
        """
        distances merupakan dictionary di mana
        key itu berupa nilai index train
        sedangkan value nya berupa nilai distance euclidean

        :param train:
        :param test:
        :return:
        """

        n_samples = test.shape[0]
        sample_range = np.arange(0,n_samples)[:, None]
        distances = {}
        predictions = []
        for i in range(len(sample_range)):
            for j in range(len(train)):
                # dist = self.euclidean_distance(train[j], test[i])
                dist = self.euclidean_distance(train.iloc[j], test.iloc[i])
                distances[j] = dist

            sorted_distances = sorted(distances.items(), key = lambda kv:(kv[1], kv[0]))

            y_pred = 0.0
            for x in range(self.k):
                index_sorted_distances = sorted_distances[x][0]
                y_pred += train.iloc[index_sorted_distances].values

            y_pred = y_pred/self.k

            predictions.extend(y_pred)

        return predictions
    #             sort distance



    """
    def predict(self, test_feature_data_point):
        _y = np.array(self._y)
        _y.astype(int)
        if _y.ndim == 1:
            _y = _y.reshape(-1,1)

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        nearest_data_point_index = self.get_neighbors(self.train_feature, test_feature_data_point, self.k)
        total_val = 0.0
        # calculate the sum of all the label values
        for index in nearest_data_point_index:
            total_val += self.train_label[index]

        return total_val/self.k
    """

def get_rmse(y, y_pred):
    '''Root Mean Square Error
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    '''
    mse = np.mean((y - y_pred)**2)
    return np.sqrt(mse)

def get_mape(y, y_pred):
    '''Mean Absolute Percent Error
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    '''
    perc_err = (100*(y - y_pred))/y
    return np.mean(abs(perc_err))


def get_accuracy(y, y_pred):
    cnt = (y == y_pred).sum()
    return round(cnt/len(y), 2)


def main():
    symbol = 'ADHI.JKT'
    stock, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
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

    # stock_numeric_close = df[distance_column_1]
    # stock_numeric_close_date = df[distance_column_2]
    # stock_numeric_close_dailyreturn = df[distance_column_3]
    # stock_numeric_date_close_dailyreturn = df[distance_column_4]
    # print(df.columns.toList())


    test_cutoff = math.floor(len(df) / 1.5)

    test = df.loc[df.index[test_cutoff:]]
    train = df.loc[df.index[1:test_cutoff]]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # test = df.loc[df.index[test_cutoff:]]
    # train = df.loc[df.index[:test_cutoff]]

    # X_train, X_test, y_train, y_test = train_test_split(df['close'], df['close_next'], test_size=0.3, shuffle=False)
    x_column = ['close']

    y_column = ['close_next']

    # a = test[x_column]

    knr = KnnRegression(6)
    knr.fit(train[x_column], train[y_column])
    predictions = knr.predict_by_myself_method(train[x_column], test[x_column])
    predictions = pd.DataFrame(np.row_stack(predictions))
    print(predictions)

    actual = train[x_column]
    print(math.sqrt(mean_squared_error(actual, predictions)))

    rmse = math.sqrt((((predictions - actual) ** 2).sum()) / len(predictions))
    print(rmse)

    trace_actual = go.Scatter(
        x=df['date'],
        y=actual['close_next'],
        name=symbol + " Actual",
        line=dict(color='red'),
        opacity=0.8

    )

    trace_predict = go.Scatter(
        x=df['date'],
        y=predictions,
        name=symbol + " Predictions",
        line=dict(color='green'),
        opacity=0.8
    )

    data_trace = [trace_actual, trace_predict]

    layout = dict(
        title=symbol + " ranges slider",
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
    py.plot(fig, filename=symbol + " stock price prediction")

if __name__ == '__main__':
    main()