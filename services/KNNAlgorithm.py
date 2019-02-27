# sumber
# https://github.com/sammanthp007/Stock-Price-Prediction-Using-KNN-Algorithm/blob/master/knnAlgorithm.py

import random, json, math, operator, datetime, csv
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib
import seaborn
from scipy import stats
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import matplotlib.pyplot as plt # import matplotlib
# from pandas.core.indexes import range
from alpha_vantage.timeseries import TimeSeries
from services.KEY import getApiKey

class KNNAlgo:

    # handle data
    # sumber
    # https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

    # split the data into a trainingdataset and testdataset in ratio of 67/33
    split = 0.67

    def loadDataset(self,filesource, split, trainingSet=[], testSet=[], content_header=[]):
        with open(filesource, 'rb') as csvfile:
            #mengembalikan reader object yg akan diiterasi per baris
            lines = csv.reader(csvfile)
            # dataset adalah list dr seluruh data, yang tiam baris adalah list
            dataset = list(lines)
            #dikurang 1 karena ingin prediksi hari selanjutnya
            for x in range(len(dataset)-1):
    #             convert the content to float
    #              dikurang 1 karena yang terakhir kolom volume
                for y in range(1, len(content_header)-1):
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])

    def readData(self, symbolstock):
        ts = TimeSeries(getApiKey(), output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol=symbolstock, outputsize='full')



    def euclideanDistance(self,instance1, instance2, length):
        distance = 0
        for x in range(1, length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    # mendapatkan k tetangga terdekat dari <array><num> testInstance diantara <array><array>
    #trainingSet
    def getNeighbors(self,trainingSet, testInstance, k):
        distance=[]
        length = len(testInstance) - 1 #dikurang 1 karena data dipecah dan test jg kelas yg sudah diketahui

        for x in range((len(trainingSet))):
            dist = self.euclideanDistance(testInstance, trainingSet[x], length)
            distance.append((trainingSet[x]), dist)

        # mengurutkan berdasarkan item yg ada pada index 1, distance
        distance.sort(key=operator.itemgetter(1))
        neighbors=[]
        for x in range(k):
            neighbors.append(distance[x][0])
        return neighbors


    # buat semua respon untuk vote di klasifikasinya, yg paling tinggi yang menang
    def getResponse(self, neighbors):
        classVotes={}
        for x in range (len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]


    # menggunakan Testset
    def getAccuracy(testSet, predictions):
        knn_correct = 0
        guess_correct=0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
               knn_correct += 1
            if random.random(0,1) == testSet[x][-1]:
                guess_correct +=1

        test = stats.chisquare([knn_correct, len(testSet)-1-knn_correct], f_exp=[guess_correct, len(testSet)-1-guess_correct])
        p = test[1]
                
        return (knn_correct/ float(len(testSet))) * 100.0, (guess_correct/float(len(testSet)-1))*100,0, p


    # menggunakan RMSD
    def getAccuracy1(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if self.RMSD(testSet[x][-1], predictions[x]) < 1:
                correct += 1
            return (correct/float(len(testSet))) * 100.0


    # root mean square deviation
    def RMSD(self, X, Y):
        return math.sqrt(pow(Y-X, 2))


    def change(self, today, yesterday):
        if today > yesterday:
            return 'up'
        return 'down'

    # def getData(self, filesource, stockcode, enddate):
    #     stock = pdData.DataReader(stockcode, )

    def predictFor(self, k, filesource, stockcode, enddate, writeagain, split):
        column_header = ["timestamp", "open", "high", "low", "yesterday close", "state change"]
        trainingSet=[]
        testSet=[]
        totalCount=0

        if writeagain:
            print("making a network request")
            self.getData(filesource, stockcode, enddate)


        self.loadDataset(filesource,split, trainingSet, testSet, column_header)

        print("Predicting for ", stockcode)
        print("Train: " + repr(len(trainingSet)))
        print("Test: " + repr(len(testSet)))
        totalCount += len(trainingSet) + len(testSet)
        print("Total: " + repr(totalCount))

    #     buat prediksi dan akurasi. return prediction and accuracy
        self.predict_and_get_accuracy(testSet, trainingSet, k)



    def predict_and_get_accuracy(self, testSet, trainingSet, k):
        predictions=[]
        for x in range(len(testSet)):
            neighbors=self.getNeighbors(trainingSet, testSet[x], k)
            result=self.getResponse(neighbors)
            predictions.append(result)

        accuracy = self.getAccuracy(testSet,predictions)
        print('Accuracy: ' + repr(accuracy)+ '%')

        return predictions, accuracy
        # gambar grafik sesuai prediksi

        #gambar grafik yang sebenarnya

    # dapat dari web
    def getData(self, filename, stockCode, enddate):
        ts = TimeSeries(key=getApiKey())
        stock_ts_data, stock_meta_data = ts.get_daily_adjusted(stockCode, outputsize='full')
        print('done making network call')

        first_time = True
        with open(filename, 'wb') as pp:
            stockwriter = csv.writer(pp)
            stocksorted = sorted(stock_ts_data.head())
            for i in stocksorted:
                new_format_date = i[:10]
                if first_time:
                    first_time = False
                    prev_closing = stock_ts_data[i]['4. close']
                    continue
                stockwriter.writerow([new_format_date] + [stock_ts_data[i]['1. open']] + [stock_ts_data[i]['2. high']] + [stock_ts_data[i]['3. low']] + [stock_ts_data[i]['4. close']] + [self.change(stock_ts_data[i]['4. close'], prev_closing)] )
                prev_closing = stock_ts_data[i]['4. close']




