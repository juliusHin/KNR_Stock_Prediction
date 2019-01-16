# sumber
# https://github.com/sammanthp007/Stock-Price-Prediction-Using-KNN-Algorithm/blob/master/knnAlgorithm.py

import random, json, math, operator, datetime, csv
import pandas_datareader.data as pdData
import pandas as pd
import numpy as np
import matplotlib
import seaborn

import matplotlib.pyplot as plt # import matplotlib
# from pandas.core.indexes import range


class KNNAlgo:

    # handle data
    # sumber
    # https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

    # split the data into a trainingdataset and testdataset in ratio of 67/33
    split = 0.67

    def loadDataset(filesource, split, trainingSet=[], testSet=[], content_header=[]):
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
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0

    # menggunakan RMSD
    def getAccuracy1(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if self.RMSD(testSet[x][-1], predictions[x]) < 1:
                correct += 1
            return (correct/float(len(testSet))) * 100.0

    def RMSD(self, X, Y):
        return math.sqrt(pow(Y-X, 2))

    def change(self, today, yesterday):
        if today > yesterday:
            return 'up'
        return 'down'