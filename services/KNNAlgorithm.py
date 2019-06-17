# sumber
# https://github.com/sammanthp007/Stock-Price-Prediction-Using-KNN-Algorithm/blob/master/knnAlgorithm.py

import numpy as np
from scipy.spatial import distance


# from pandas.core.indexes import range

class KnnAlgo(object):
    def __init__(self, k):
        self.k = k

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        return self


class KnnRegression(KnnAlgo):
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
        sample_range = np.arange(0, n_samples)[:, None]
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

            sorted_distances = sorted(distances.items(), key=lambda kv: (kv[1], kv[0]))

            y_pred = 0.0
            for x in range(self.k):
                index_sorted_distances = sorted_distances[x][0]
                y_pred += train.iloc[index_sorted_distances].values

            y_pred = y_pred / self.k

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
