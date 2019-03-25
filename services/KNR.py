import math, operator
from scipy.spatial import distance

class KNR(object):
    def __init__(self, x, y, k , weighted=False):
        assert (k <=len(x)), "k tidak bisa lebih besar dari panjang training_set"
        self.__x = x
        self.__y = y
        self.__k = k
        self.__weighted = weighted

    @staticmethod
    def __euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def gaussian(dist, sigma=1):
        return 1. / (math.sqrt(2. * math.pi) * sigma) * math.exp(-dist ** 2 / (2 * sigma ** 2))

    def y_value_base_neighbors(self,k, y=0):
        i=1
        for i in k+1:
            y+=y[i]
        return y

    def weight_value_calc(self, k, x_query_point, p_i):
        distance_euclidean = distance.euclidean(x_query_point, p_i)
        weight_k = 0
        i=1
        for i in k+1:
            weight_k += distance.euclidean(x_query_point, p_i[i])
        weight = math.exp(distance_euclidean*-1) / weight_k
        return weight

    # prediksi untuk y
    def y_predict(self, k, X_0, X_i, y_i):
        y_predicted = 0
        i=1
        for i in k+1:
            y_predicted += self.weight_value_calc(k, X_0, X_i) * self.y_value_base_neighbors(k, y_i)
        return y_predicted
