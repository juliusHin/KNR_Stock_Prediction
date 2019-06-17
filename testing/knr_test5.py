import math

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
from alpha_vantage.timeseries import TimeSeries
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor

from services.KEY import getApiKey

plotly.tools.set_credentials_file(username='junhin04', api_key='c0YKlYKB4vqlUGfaEDNO')

symbol = "ADHI.JKT"

ts = TimeSeries(key=getApiKey(), output_format='pandas')

stock, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')
stock = stock.sort_values('date')

open_price = stock['1. open'].values
low = stock['3. low'].values
high = stock['2. high'].values
close = stock['4. close'].values
sorted_date = stock.index.get_level_values(level='date')

stock_numpy_format = np.stack((sorted_date, open_price, low
                               ,high, close), axis=1)
df = pd.DataFrame(stock_numpy_format, columns=['date', 'open', 'low', 'high', 'close'])

df = df[df['open']>0]
df = df[(df['date'] >= "2016-01-01") & (df['date'] <= "2018-12-31")]
df = df.reset_index(drop=True)

df['close_next'] = df['close'].shift(-1)
df['daily_return'] = df['close'].pct_change(1)
df.fillna(0, inplace=True)

distance_column_1 = ['close']
distance_column_2 = ['date', 'close']
distance_column_3 = ['close', 'daily_return']
distance_column_4 = ['date', 'close', 'daily_return']

stock_numeric_close = df[distance_column_1]
stock_numeric_close_date = df[distance_column_2]
stock_numeric_close_dailyreturn = df[distance_column_3]
stock_numeric_date_close_dailyreturn = df[distance_column_4]

# stock_normalized = (stock_numeric_close_dailyreturn - stock_numeric_close_dailyreturn.mean()) / stock_numeric_close_dailyreturn.std()
# stock_normalized = (stock_numeric_close_date - stock_numeric_close_date.mean()) / stock_numeric_close_date.std()
stock_normalized = (stock_numeric_close - stock_numeric_close.mean()) / stock_numeric_close.std()

# stock_normalized.fillna(0, inplace= True)
# date_normalized = stock_normalized[df["date"] == "2016-12-29"]
# euclidean_distances = stock_normalized.apply(lambda row: distance.euclidean(row, date_normalized) , axis=1)
# distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx":euclidean_distances.index})
# distance_frame.sort_values("dist", inplace=True)
# second_smallest = distance_frame.iloc[1]["idx"]
# most_similar_to_date = df.loc[int(second_smallest)]["date"]

test_cutoff = math.floor(len(df) / 1.5)
test = df.loc[df.index[test_cutoff:]]
train = df.loc[df.index[1:test_cutoff]]

# x_column = ['close', 'daily_return']

# x_column = ['date', 'close']

x_column = ['close']

y_column = ['close_next']

# X_train, X_test, y_train, y_test = train_test_split(df['close'].reshape(-1, 1), df['close_next'].reshape(-1,1), test_size=0.3, shuffle=False)


knn = KNeighborsRegressor(n_neighbors=6, weights='distance', metric='euclidean')
knn.fit(train[x_column], train[y_column])
predictions = knn.predict(test[x_column])

# coba ubah index nya supaya mulai dari index test
# predictions.index = test.index.get_level_values(0).values

# knn.fit(X_train, y_train)
# predictions = knn.predict(X_test)

actual = test[y_column]
# actual = pd.DataFrame(test[y_column], test.index.get_level_values(0).values)
# actual.index =
# print(knn.score(test[x_column], test[y_column]))
# print(metrics.accuracy_score(actual,predictions))

print(math.sqrt(metrics.mean_squared_error(actual, predictions)))

rmse = math.sqrt((((predictions - actual) ** 2).sum()) / len(predictions))
print(rmse)
r_square = metrics.r2_score(actual, predictions)
print(r_square)
# print(knn.score(predictions,actual))


'''
plt.plot(predictions, 'r')
plt.plot(actual, 'g')
plt.xlabel('Day in Future', fontsize=30)
plt.ylabel('Price', fontsize=30)
red_patch = mpatches.Patch(color='red', label='Predicted Price')
green_patch = mpatches.Patch(color='green', label='Actual Price')
plt.legend(handles=[red_patch])
plt.legend(handles=[red_patch, green_patch])
plt.show()
'''

trace_actual = go.Scatter(
    x=df['date'][test_cutoff:],
    y = actual['close_next'],
    name= symbol + " Actual",
    line=dict(color='red'),
    opacity= 0.8

)

trace_predict = go.Scatter(
    x=df['date'][test_cutoff:],
    y = predictions,
    name= symbol + " Predictions",
    line=dict(color='green'),
    opacity=0.8
)

data_trace = [trace_actual, trace_predict]

layout = dict(
    title=symbol + " ranges slider (" + str(r_square) + ")",
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

fig = dict(data = data_trace, layout=layout)
py.plot(fig, filename = symbol + " stock price prediction")