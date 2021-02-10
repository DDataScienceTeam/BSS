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
