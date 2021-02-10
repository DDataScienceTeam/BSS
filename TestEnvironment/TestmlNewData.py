import pickle
import pyodbc
from datetime import date
from datetime import datetime
from datetime import timedelta as td
import pandas as pd
import numpy as np


def gmmNewData(peaksByPeaks,gmmDf, startDate, endDate, durationStr):
    peaksByPeaksFilt = peaksByPeaks.filter(peaksByPeaks.timestamp > startDate)
    peaksByPeaksFilt2 = peaksByPeaksFilt.filter( peaksByPeaksFilt.timestamp < endDate)
    peaksByPeaksPdf = peaksByPeaksFilt2.toPandas()
    print(peaksByPeaksPdf.shape)
    gmmPdf = gmmDf.toPandas()
    #Unpack order for gmmList: clf, scaler, liklihood_thresh threshold
    mlDonutPdf = pd.DataFrame([], columns = ['deviceID', 'timeRecord','durationStr', 'mlGood', 'mlWarning', 'mlBad', 'mlScore'])
    for j, (name, group) in enumerate(peaksByPeaksPdf.groupby('deviceID')):
        descript = np.array([name, endDate.date(), durationStr])
        print(name, group.shape[0])
        gmmCol = name.replace(" ", "")
        gmmPickle = gmmPdf[gmmCol].iloc[0]
        gmmList = pickle.loads(gmmPickle)
    #     print(gmmList)
        dataTest = group[['frequency', 'magnitude']].values
        #Load in the relevant gmm Data
        percentOutlier = gmmPredict(dataTest, gmmList[0], gmmList[1], gmmList[2], gmmList[3])
        
        if group.shape[0] < 4:
            percentOutlier = 0
            
        if percentOutlier > 80:
            mlArray = np.array([0,0,1])
        elif percentOutlier > 60:
            mlArray = np.array([0,1,0])
        else:
            mlArray = np.array([1,0,0])

        row = np.append(descript, mlArray)
        row = np.append(row, np.array([int(percentOutlier/10)]))
#         print(row)
        rowDf = pd.DataFrame([row], columns = ['deviceID', 'timeRecord','durationStr', 'mlGood', 'mlWarning', 'mlBad', 'mlScore'])
        mlDonutPdf = mlDonutPdf.append(rowDf)
        print(percentOutlier)
#     mlDonutDf = spark.createDataFrame(mlDonutPdf)
    return mlDonutPdf