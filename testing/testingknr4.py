'''
to Anyone who read this or use this application
Thanks to
naman05rathi

who gave the simplest k neighbors regression to all people include me
https://github.com/naman05rathi/JNU_StockPrice/tree/185a8d8ef7f9d7920cc5b9926ca67628772cdb33

'''

import csv, sys
import random
import math
import operator
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import pdist,squareform
from numpy.random import permutation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsRegressor
from alpha_vantage.timeseries import TimeSeries
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from services.KEY import getApiKey
from numpy.random import permutation
from sklearn.metrics import accuracy_score

plotly.tools.set_credentials_file(username='junhin04', api_key='c0YKlYKB4vqlUGfaEDNO')

symbol = "ADHI.JKT"
ts = TimeSeries(key=getApiKey(), output_format='csv')

file_path = "../dataset/daily_adjusted_ADHI.JKT.csv"

with open(file_path,'r') as csvfile:
    # data_csv, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    stock = pd.read_csv(file_path)
    # stock, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    # reader = csv.reader(open(file_path), delimiter=",")
    stock = stock[stock['open'] > 0]
    stock = stock[(stock['timestamp'] >= '2016-01-01') & (stock['timestamp'] <= '2018-12-31')]
    stock = stock.sort_values('timestamp')
    stock = stock.reset_index(drop=True)
    stock['close_next'] = stock['close'].shift(-1)
    stock['daily_return'] = stock['close'].pct_change(1)
    # stock[np.isnan(stock['daily_return'])] = 0
    stock.fillna(0, inplace=True)
    print(stock)
    stock.to_csv('ADHI.JKT_full.csv',mode='w', header=True)


selected_date = stock.iloc[0]
distance_column = ['close', 'daily_return']

def euclidean_distance(row):
    inner_value = 0
    for k in distance_column:
        inner_value += (row[k] - selected_date[k])**2
    return math.sqrt(inner_value)

date_distance = stock.apply(euclidean_distance, axis=1)
stock_numeric = stock[distance_column]
stock_normalized = (stock_numeric - stock_numeric.mean()) / stock_numeric.std()
stock_normalized.fillna(0, inplace= True)
date_normalized = stock_normalized[stock['timestamp'] == "2018-12-28"]
euclidean_distance = stock_normalized.apply(lambda row:distance.euclidean(row, date_normalized), axis=1)
distance_frame = pd.DataFrame(data={"dist": euclidean_distance, "idx": euclidean_distance.index})
distance_frame.sort_values("dist", inplace=True)
second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_date = stock.loc[int(second_smallest)]['timestamp']

test_cutoff = math.floor(len(stock)/3)
test = stock.loc[stock.index[1:test_cutoff]]
train = stock.loc[stock.index[test_cutoff:]]
x_column = ['close', 'daily_return']
y_column = ['close_next']


knn = KNeighborsRegressor (n_neighbors = 3)
knn.fit(train[x_column], train[y_column])
predictions = knn.predict(test[x_column])
# score_preidict = knn.score(test[x_column], test[y_column])
# print ('score %f with neighbors %d' %())
# print(score_preidict)
actual = test[y_column]


plt.plot(predictions, 'r')
plt.plot(actual, 'g')
plt.xlabel('Day in Future', fontsize=30)
plt.ylabel('Price', fontsize=30)
red_patch = mpatches.Patch(color='red', label='Predicted Price')
green_patch = mpatches.Patch(color='green', label='Actual Price')
plt.legend(handles=[red_patch])
plt.legend(handles=[red_patch, green_patch])
plt.show()


mse = (((predictions - actual) ** 2).sum())/len(predictions)
print (mse)