import math
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import pandas as pd
from datetime import datetime, timedelta


def ParseData(path):
    #Read the csv file into a dataframe
    df = pd.read_csv(path)
    #Get the date strings from the date column
    dateStr = df['Date'].values
    D = np.zeros(dateStr.shape)
    #Convert all date strings to a numeric value
    for i, j in enumerate(dateStr):
        #Date strings are of the form month/day/year
        D[i] = datetime.strptime(j, '%m/%d/%Y').timestamp()
    #Add the newly parsed column to the dataframe
    df['Timestamp'] = D
    #Remove any unused columns (axis = 1 specifies fields are columns)
    return df.drop('Date', axis = 1)

def KNNRegression(NNeighbors, D, interval=1):
    #Create arrays for features (timestamp) and targets (closing price)
    knnIn = np.zeros((len(D.Timestamp),1))
    knnTarget = np.zeros((len(D.Timestamp),1))
    #Populate arrays with data from the input matrix
    for i in range (0, len(D.Timestamp)):
        knnIn[i] = D.Timestamp[i]
        knnTarget[i] = D.Close[i]
    #Create the prediction array (What will be tested against the closing price of a day)
    knnPredict = np.zeros((len(D.Timestamp),1))
    #Don't have the data to predict for the first NNeighbors days (because prediction requires at least NNeighbors previous points)
    knnPredict[0:NNeighbors+1] = 0
    #Create the regressor using a distance-weightedd KNN function
    R = KNeighborsRegressor(NNeighbors, 'distance')
    for i in range(NNeighbors+1,len(D.Timestamp)):
        #Fit the regressor to the current window (past NNeighbor points and their closing value targets)
        R.fit(knnIn[(i-(NNeighbors+1)):i-1].reshape(-1, 1), knnTarget[(i-(NNeighbors+1)):i-1])
        #Make a prediction for the current timestamp
        knnPredict[i] = R.predict(knnIn[i].reshape(-1, 1))
    Err = 0
    for i in range(NNeighbors+1, len(D.Timestamp)):
        #Compute the total mean absolute error percentage
        Err += math.sqrt(((D.close[i] - knnPredict[i])/D.close[i])**2)
    #Divide summed mean absolute error percentage by number of points considered
    Err = Err / (len(D.Timestamp) - NNeighbors)
    print(Err)



def main():
    path = 'https://raw.githubusercontent.com/theReveler/AIFinnaFinance/665d027a311d28c35199ab36dcec1342defa7361/AMZN.csv'
    D = ParseData(path)

    for i in range(1, 13, 2):
        print(i)
        KNNRegression(i, D)

if __name__ == '__main__':
    main()