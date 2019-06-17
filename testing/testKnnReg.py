import math

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
from alpha_vantage.timeseries import TimeSeries
from scipy.spatial import distance
# from numpy.core.tests.test_mem_overlap import xrange
# from scipy._lib.six import xrange
from sklearn.metrics import mean_squared_error, r2_score

# you can get apiKey from AlphaVantage.co
from services.KEY import getApiKey

# from services.KNNAlgorithm import KNNAlgo as knna
# from scikit learn SKLearn


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

    def distance_weighted(self, sorted_distances):

        """
        distance Weighted

        W(x,p_i) = exp(-Distance(x,p_i)) / sum_from_i=1_to_k exp(-Distance(x,p_i))

        """
        # matches = [(1, y) for d, y in distances if d == 0]
        # return matches if matches else [(1/d, y) for d, y in distances]
        weighted_dist = []
        weighted_dist_float = np.array(weighted_dist, dtype=np.float128)
        sum_dist_k = 0.0
        for i in range(self.k):
            sum_dist_k += math.exp(sorted_distances[i][1] * -1)

        for x in range(self.k):
            dist = math.exp(sorted_distances[x][1] * -1) / sum_dist_k
            weighted_dist.append(dist)
            # weighted_dist_float = np.append(weighted_dist_float, dist, axis=0)
            # weighted_dist_float.ravel()

        return weighted_dist

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

        :param train:;
        :param test:
        :return:
        """

        n_samples = test.shape[0]
        sample_range = np.arange(0,n_samples)[:, None]
        distances = {}
        predictions = []
        # distances = distance.euclidean(train, test)
        for i in range(len(sample_range)):
            for j in range(len(train)):
                # dist = self.euclidean_distance(train[j], test[i])
                # dist = self.euclidean_distance(train.iloc[j], test.iloc[i])
                dist = distance.euclidean(train.iloc[j], test.iloc[i])
                distances[j] = dist

            # distances = distance.euclidean(train, test[i])

            sorted_distances = sorted(distances.items(), key = lambda kv:(kv[1], kv[0]))

            y_pred = 0.0
            for x in range(self.k):
                index_sorted_distances = sorted_distances[x][0]
                y_pred += train.iloc[index_sorted_distances].values

            y_pred = y_pred/self.k

            predictions.extend(y_pred)

        return predictions
    #             sort distance

    def predict_by_myself_method_weighted_distance(self, train, test):
        """
                distance Weighted Regression

                y_predictions = sum_from_i=1_to_k W (X_0, X_i)y_i

        """

        n_samples = test.shape[0]
        sample_range = np.arange(0, n_samples)[:, None]
        distances = {}
        predictions = []
        for i in range(len(sample_range)):
            for j in range(len(train)):
                # dist = self.euclidean_distance(train[j], test[i])
                dist = self.euclidean_distance(train.iloc[j], test.iloc[i])
                distances[j] = dist

            sorted_distances = sorted(distances.items(), key=lambda kv: (kv[1], kv[0]))
            weighted_distances = self.distance_weighted(sorted_distances)
            y_pred = 0.0
            for x in range(self.k):
                index_sorted_distances = sorted_distances[x][0]
                # value_y = train.iloc[index_sorted_distances][0]
                # weight_dist_value = weighted_distances[x]
                y_pred += train.iloc[index_sorted_distances].values * weighted_distances[x]

            y_pred = y_pred / self.k

            predictions.extend(y_pred)

        return predictions


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
    symbol = 'WSKT.JKT'
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

    # reset index untuk lebih mudah dimapping ke dalam dictionary
    train_reset_index = train.reset_index(drop=True)
    test_reset_index = test.reset_index(drop=True)

    # test = df.loc[df.index[test_cutoff:]]
    # train = df.loc[df.index[:test_cutoff]]

    # X_train, X_test, y_train, y_test = train_test_split(df['close'], df['close_next'], test_size=0.3, shuffle=False)
    x_column = ['close']

    y_column = ['close_next']

    # a = test[x_column]

    knr = KnnRegression(6)
    knr.fit(train_reset_index[x_column], train_reset_index[y_column])
    predictions = knr.predict_by_myself_method(train_reset_index[x_column], test_reset_index[x_column])
    # predictions = knr.predict_by_myself_method_weighted_distance(train[x_column], test[x_column])

    # convert to dataframe dan indexnya diubah supaya mengikuti index dari index test (untuk index tanggal)
    predictions = pd.DataFrame(np.row_stack(predictions), index=test.index.get_level_values(0).values)

    # predictions = predictions
    print(predictions)

    # predictions reindex
    # predictions = predictions.reindex(index=range(test[x_column]))
    actual = test[y_column]
    # actual.index = test.index.get_level_values(0).values

    print(math.sqrt(mean_squared_error(actual['close_next'], predictions[0])))

    r_square = r2_score(actual['close_next'], predictions[0])
    print(r_square)
    # rmse = math.sqrt((((predictions - actual) ** 2).sum()) / len(predictions))
    # print(rmse)

    trace_actual = go.Scatter(
        x=df['date'][test_cutoff:],
        y=actual['close_next'],
        name=symbol + " Actual",
        line=dict(color='red'),
        opacity=0.8

    )

    trace_predict = go.Scatter(
        x=df['date'][test_cutoff:],
        y=predictions[0],
        name=symbol + " Predictions",
        line=dict(color='green'),
        opacity=0.8
    )

    data_trace = [trace_actual, trace_predict]

    layout = dict(
        title=symbol + " ranges slider (" + str(r_square) + ")",
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