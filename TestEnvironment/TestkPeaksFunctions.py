# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:36:56 2020

@author: Harry Bowman
"""


import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.svm import OneClassSVM
#import scipy.fft
import pandas as pd
#HEY
#%%FFT Function
def getFFT(sample, axisLen = 2048):
        x_fft_ = fft(sample, axisLen)
        x_fft = 2.0 / math.sqrt(axisLen) * np.abs(x_fft_[0:axisLen//2])
        return x_fft

#%% smoothing function
def movingAvCust(x, w = 6, ss = 1):
    smoothedX = []
    for i in np.arange(w/2, len(x)-w/2, ss):
        val = np.mean(x[int(i-w/2):int(i+w/2)])
        smoothedX.append(val)
    return np.array(smoothedX)

#%%k peaks finder
def kPeaks(wave, numPeaks = 0, width = 20, minProminence = 1, minHeightDivider = 5):
    
    #get peaks
    peaks, prop = find_peaks(wave, distance=width, prominence = (minProminence, None))
    if len(peaks) != 0:
        prom = prop['prominences']
        
        #Get max height and threshold out low peaks
        maxHeight= np.max(wave[peaks])
    #    print(maxHeight)
        minNormHeight = maxHeight/minHeightDivider
        goodPeaks = np.where(wave[peaks]> minNormHeight)
        peaks = peaks[goodPeaks]
        prom = prom[goodPeaks]
        print(type(peaks))

            # Find top numPeaks from this based on prominence unless numPeaks = 0
        if numPeaks != 0:
            topPeaks = np.sort(peaks[np.argsort(prom)[-numPeaks:]])
            topProm = []
            for i in topPeaks:
                topProm.append(prom[peaks.tolist().index(i)])
            peaks = topPeaks
            prom = topProm

    else:
        peaks = []
        prom = []
        
    return peaks, prom

def bucketData(specPdf, numBuckets = 32, aggType = 'max', spectrumLen = 512, colToDrop = ['deviceID', 'timestamp']):
    #Function description: takes in spectrum data, returns bucketed data using numBuckets as the number of buckets and the aggType determines how they are aggregated
    #Returns spark dataframe with result
    bucketWidth = int(spectrumLen/numBuckets)

    #For each deviceID, calculate bucket values and append into one dataframe
    for j,(name,group) in enumerate(specPdf.groupby('deviceID')):
        bucketMatrix = np.array([]) #for this one ID
        timestamp = group['timestamp'].values
        data = group.drop(colToDrop, axis = 1) #Just spectrum values
        for row in range(group.shape[0]): #For each spectrum reading
            singleRow = data.iloc[row]
            bucketList = []
            for i in range(0,spectrumLen, bucketWidth): #For each bucket in the one reading
                if aggType == 'mean':
                    bucketVal = np.mean(singleRow[i:i+bucketWidth])
                else:
                    bucketVal = np.max(singleRow[i:i+bucketWidth])
                bucketList.append(bucketVal)
            bucketArray = np.array([bucketList])

            if row == 0:
                bucketMatrix = bucketArray
            else:
                bucketMatrix = np.concatenate((bucketMatrix, bucketArray), axis = 0)
        
        #Join the deviceID and time info onto the bucket information
        nameArray = np.transpose(np.array([[name for i in range(bucketMatrix.shape[0])]]))
        npTime = np.transpose(np.array([timestamp]))
        nameTimeArray = np.concatenate((nameArray, npTime), axis=1)
        bucketMatrix = np.concatenate((nameTimeArray, bucketMatrix), axis=1)
        
        #Join matrix onto previous ones to create one large matrix
        if j ==0:
            fullMatrix = bucketMatrix
        else:
            fullMatrix = np.concatenate((fullMatrix, bucketMatrix), axis = 0)
    bucketPdf = pd.DataFrame(fullMatrix, columns = ['deviceID', 'timestamp']+[str(i) for i in range(numBuckets)])
    bucketDf = spark.createDataFrame(bucketPdf)
    return bucketDf

###############################################
#Functions to generate the peaks data by peaks
###############################################
def peaksData(specPdf, widthK = 20, width = 10, stepSize = 5, promSF = 5, spectrumLen = 512):

    outputList = []
    outputNames = []

    for j,(name,group) in enumerate(specPdf.groupby('deviceID')):
        print(name)
        outputNames.append(name)
        freqVector = np.array([])
        magVector = np.array([])
        timeVector = np.array([])
        for row in range(group.shape[0]):
            data = group.drop(['deviceID', 'timestamp'], axis = 1)
            x_fft = data.iloc[row].astype(float)
            smoothed = movingAvCust(x_fft,w=width, ss=stepSize)
            smoothed = np.insert(smoothed, 0, 0)
            minPromVal = (np.max(smoothed))/promSF
            topPeaks, prom= kPeaks(smoothed, numPeaks = 6, width = width, minProminence = minPromVal)
            mag = smoothed[topPeaks]
            magVector = np.concatenate((magVector, mag))
            freqVector = np.concatenate((freqVector, topPeaks))
            timeList = [group.iloc[row]['timestamp'] for i in range(len(mag))]
            timeVector = np.concatenate((timeVector, np.array(timeList)))
        freqVector  =freqVector * 100
        data= np.column_stack((magVector, freqVector))
        output = pd.DataFrame(data, columns = ['magnitude', 'frequency'])
        print('otuptu = ',len(magVector))
        print('timeVector = ',len(timeVector))
        output['timestamp'] = timeVector
        output['deviceID'] = name
        outputDf = spark.createDataFrame(output)
        outputList.append(outputDf)
        if j == 0:
            finalPdf = output
        else:
            finalPdf = finalPdf.append(output)
        

    finalPdf = finalPdf.replace(['a84041000181a5f1', 'a84041000181a5ca','a84041000181a5c3','a84041000181cc92','a84041000181cc98','a84041000181cc9e'],['Blower 1', 'Blower 2', 'Blower 3', 'Blower 4', 'Blower 5', 'Blower 6'])
    
    # add in Invalid boolean, default 0, meaning valid
    finalPdf['Invalid'] = 0

    #Drop the rows of small magnitudes
    finalPdfNosmall = finalPdf.loc[finalPdf.magnitude > 0.2]

    finalDf = spark.createDataFrame(finalPdfNosmall)

    return finalDf
