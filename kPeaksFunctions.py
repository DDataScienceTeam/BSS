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
def kPeaks(wave, numPeaks = 3, width = 20, minProminence = 1, minHeightDivider = 5):
    
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
        
        
        #Find top 3 from this based on prominence
    #    topPeaks = np.sort(peaks[np.argsort(prom)[-numPeaks:]])
    else:
        peaks = []
        prom = []
        
    return peaks, prom

def bucketData(specPdf, numBuckets = 32, aggType = 'max', spectrumLen = 512):
    #Function description: takes in spectrum data, returns bucketed data using numBuckets as the number of buckets and the aggType determines how they are aggregated
    #Returns spark dataframe with result
    bucketWidth = int(spectrumLen/numBuckets)

    #For each deviceID, calculate bucket values and append into one dataframe
    for j,(name,group) in enumerate(specPdf.groupby('deviceID')):
        bucketMatrix = np.array([]) #for this one ID
        timestamp = group['timestamp'].values
        data = group.drop(['deviceID', 'timestamp'], axis = 1) #Just spectrum values
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
