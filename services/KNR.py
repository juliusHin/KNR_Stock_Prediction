import math, operator
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from alpha_vantage.timeseries import TimeSeries
from services.KEY import getApiKey
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from scipy.spatial import distance
from random import randint, seed, random
from sortedcontainers import SortedList
from scipy.sparse import issparse


# sumber
"""
https://github.com/matthew-mcateer/Lucretius/blob/8e41c767dc63a97397fe126434f72d322123fc77/Lucretius/k-nearest-neighbors.py

https://stathwang.github.io/k-nearest-neighbors-from-scratch-in-python.html

"""

class KNN_Regression(object):
    def __init__(self,k):
        self.__k = k

    def fit(self, X, y):
        self.__X = X
        self.__y = y


    def predict(self, X):
        predictions = []
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList(iterable=self.k)
            for j, xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))

            y[i] = np.mean([pred[1] for pred in sl])

        return y


    def score(self, X, Y):
        P = self.predict(X)
        return math.sqrt((((P-Y) **2).sum() / len(P) ))

    '''
    sumber https://github.com/amallia/kNN/blob/master/kNN.py
    '''

    def predict2(self, test_set, distance_list, data_set):
        predictions = []
        distance = distance_list
        v=0
        # total_weight = 0
        for i in range (self.__k):
            v += distance[i][0]

        predictions.append(v/self.__k)
        return predictions

    '''
    sumber
    https://github.com/DavidCico/Self-implementation-of-KNN-algorithm/blob/master/My_KNN.py
    '''

    # Euclidean Distance
    def EuclideanDistance(self, instance1, instance2, length):
        distance = 0
        for i in range(length):
            distance += pow(instance2[i] - instance1[i], 2)

        return math.sqrt(distance)

    # Get neighbors
    def getNeighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance) - 1

        for i in range(len(trainingSet)):
            dist = self.EuclideanDistance(testInstance, trainingSet.iloc[i], length)

            '''
                        
            if distancetype == "Euclidean":
                dist = DistanceType.EuclideanDistance(testInstance, trainingSet[i], length)
            elif distancetype == "Manhattan":
                dist = DistanceType.ManhattanDistance(testInstance, trainingSet[i], length)
            else:
                dist = DistanceType.MinkowskiDistance(testInstance, trainingSet[i], length, *args)
            
            '''

            distances.append((trainingSet.iloc[i], dist))

        distances.sort(key=operator.itemgetter(1))
        # return distances
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])

        return neighbors

    # Regression by taking mean from neighbors (Regression problem)
    def getRegression(self, neighbors):
        output_values = [row[-1] for row in neighbors]

        return sum(output_values) / float(len(output_values))



    def knn_regressor(x_train, y_train, x_test, k):
        regressor_predictions = []

        for datapoint in x_test:
            distance_result = []

            #         hitung jarak untuk setiap training set datapoint
            for index, vector in enumerate(x_train):
                distance_result.append(distance.euclidean(datapoint, vector), y_train[index])

            distance_result.sort()

            #        agregrat nilai neighbor
            average = 0
            for i in range(k):
                average += distance_result[i][1]

            average = average / float(k)
            regressor_predictions.append(average)

        return regressor_predictions


    # sumber,
    #https://github.com/scikit-learn/scikit-learn/blob/7b136e92acf49d46251479b75c88cba632de1937/sklearn/neighbors/regression.py


    def predict(self, X):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of int, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        """
        if issparse(X) and self.metric == 'precomputed':
            raise ValueError(
                "Sparse matrices not supported for prediction with "
                "precomputed kernels. Densify your matrix."
            )
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

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

        return y_pred


    # def distance_weighted(self, distance_list, k):
    #     exp_distance = []
    #     weight=[]
    #     for i in enumerate(distance_list):
    #         exp_distance.append(math.exp(distance_list[i]*-1))
    #
    #     for i  in k:
