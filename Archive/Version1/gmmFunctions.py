# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:11:13 2020

@author: Harry Bowman
"""
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import MinMaxScaler as dataScaler
from sklearn.mixture import GaussianMixture as GM
from sklearn.metrics import silhouette_score as silScore
from sklearn.metrics import davies_bouldin_score as dbScore
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from scipy.special import logsumexp


def gmmFit(data, numMixRange = 10, scoreToUse = 'sil', threshold=0.9, mixToUse = 0):
    scaler = dataScaler().fit(data)
    X = scaler.transform(data)
    likelihood_threshold = np.quantile(scaler.transform(data), 1 - threshold)
    scoreList = []
    #scoreToUse = 'db' #or sil, or nothing for my old bad way 
    if scoreToUse == 'db':
        bestScore = -10000
    else:
        bestScore= 10000
    if mixToUse == 0:
        for i in np.arange(2,numMixRange+1):
            gmm = GM(i).fit(X)
            labels = gmm.predict(X)
    
            if scoreToUse == 'sil':
                currentScore = silScore(X, labels)
                if currentScore < bestScore:
                    bestScore = currentScore
                    bestGmm  = gmm
                    bestMix = i
                    
            elif scoreToUse == 'db':
                currentScore = dbScore(X, labels)
                if currentScore > bestScore:
                    bestScore = currentScore
                    bestGmm  = gmm
                    bestMix = i
            
            else:
                currentScore = gmm.bic(X) + (20*(i**1.5/numMix))
                if currentScore > bestScore:
                    bestScore = currentScore
                    bestGmm  = gmm
                    bestMix = i
            
            scoreList.append(currentScore)
    else:
        bestGmm = GM(mixToUse).fit(X)
        bestScore = 8
        bestMix = mixToUse

    clf = bestGmm

    #Get decision Boundaries with scatter plotting
    minMax = np.array([[0,0], [10240, 6]])
    minMaxScaled = scaler.transform(minMax)
    meshSize = 1000
    xx, yy = np.meshgrid(np.linspace(minMaxScaled[0,0], minMaxScaled[1,0], meshSize), np.linspace(minMaxScaled[0,1], minMaxScaled[1,1], meshSize))
    U = np.concatenate([xx[...,None],yy[...,None]],2).reshape([-1,2])
    prob_U = -clf.score_samples(U)+likelihood_threshold
    uPlot = prob_U.reshape([meshSize,meshSize])
    mask = (uPlot<0).astype(np.int)
    scatterCoord = mask.nonzero()
    scatterCoordFreq = scatterCoord[1]
    scatterCoordMag = scatterCoord[0]
    return clf, bestMix, scaler, bestScore, scatterCoordMag, scatterCoordFreq
  
    
    
    
    
def gmmPredict(data, clf, scaler, likelihood_threshold, titleVal = 'PROVIDE TITLE VAL', bestMix = 88, bestScore = 0, plot = False, percentAnom = 1):
    ###############################
    ###PREDICT ON THE DATA
    ###############################
    
    X = scaler.transform(data)
    score_samples = clf.score_samples(X)
    prob = -clf.score_samples(X) + likelihood_threshold
    if percentAnom:
        numOutlier = len(prob[prob>0])
        percentOutlier = numOutlier/len(data) * 100
    else:
        percentOutlier = np.mean(prob,0)
    #print('a')
    if plot:
        print('b')
        ######################################
        #PREDICT ON THE MESH
        ######################################
        #Generate background for plotting against
        freq = X[:,0]
        mag = X[:,1]
        print(mag.min())
        print(mag.max())
        meshSize = 200
        plt.figure()
        ax1 = plt.subplot(1,2,1)
        plt.scatter(freq,mag,c=prob>0,s=8, alpha = 0.2)
        xx, yy = np.meshgrid(np.linspace(freq.min(), freq.max(), meshSize), np.linspace(mag.min(), mag.max(), meshSize))
        U = np.concatenate([xx[...,None],yy[...,None]],2).reshape([-1,2])

        
        #get mesh probabilities
        score_samplesMesh = clf.score_samples(U)
        prob_U = -clf.score_samples(U)+likelihood_threshold
        ax2= plt.subplot(1,2,2, sharex = ax1, sharey = ax1)
        # plt.scatter(U[:,0],U[:,1],c=prob_U>0,s=8)
        uPlot = prob_U.reshape([meshSize,meshSize])
        print(prob_U.shape)
        #Plot decision boundaries
        # CS = ax1.contour(uPlot, [0], origin='lower', cmap='flag',linewidths=2, extent = [np.min(xx), np.max(xx), np.min(yy), np.max(yy)])
        uPlotUnnorm = (scaler.inverse_transform(prob_U)).reshape([meshSize,meshSize])
        CS = ax2.contourf(uPlotUnnorm, [-1e5,0], origin='lower', extent = [np.min(xx), np.max(xx), np.min(yy), np.max(yy)], colors='green', alpha = 0.2)
        plt.colorbar()
        titleString = titleVal + 'numMix = ' + str(bestMix) + 'score = ' + str(bestScore)
        plt.suptitle(titleString)
        

    return percentOutlier
    
        
        
      
def anomVal(loss, timeData = 'k', anomThresh = 15, anomThreshNeg = 2, numStd = 4, maxVal = 1,title = '1',index=1, plot = 0, percentTrain = 10, percentVal = 20):
    trainLen = int(len(loss)*percentTrain/100)
    valLen = int(len(loss)*percentVal/100)
    lossTrain = loss[:trainLen]
    lossVal = loss[trainLen:valLen]
    lossTest = loss[valLen:]
   # print(valLen+trainLen)
    # print('loss = ', loss)
    #generate Threshold
    lossThresh = np.mean(lossVal) + numStd*np.std(lossVal)+5
    #iterate through lossTest
    anomCount = 0
    anomCountNeg = 0
    anomFound = 0
    anomDetectIteration = len(loss)-1
    anomFalseDetectIteration = []
    falseAnomalyFound = 0 #Bool for if anom detected, and then lost
    for j, lossSingle in enumerate(loss):
        #Condition for checking for anomaly
        if anomFound == 0:
            if lossSingle > lossThresh:
                # print(j,  anomThresh - anomCount)
                anomCount+=1
                #If sufficent anomalies have occured
                if anomCount > anomThresh:
                    anomDetectIteration = j-anomThresh
                    anomFound = 1
                    print('anomalyFound')
            else:
                anomCount = 0
        
        #Condition for checking if stays anomalous
        else:
            if lossSingle < lossThresh:
                anomCountNeg+=1
                # print(anomThreshNeg - anomCountNeg)

                if anomCountNeg > anomThreshNeg:

                    falseAnomalyFound =1
                    anomFalseDetectIteration.append(j-anomThreshNeg)
                    print('anomFalseDetectIteration = ', j-anomThreshNeg)
                    anomFound = 0
            else:
                anomCountNeg = 0
    # print(loss)
    print('anomFalseDetectIteration for',title,' = ', anomFalseDetectIteration)
    if plot:
        colorList = ['g', 'b', 'r', 'y', 'k', 'c', 'm', 'orange', 'wheat']
        if len(timeData) == 1:
            # print(loss)
            plt.plot(loss, label = title, color = colorList[int(title)-1])
            plt.legend()
            plt.plot([lossThresh for i in range(len(loss))], label = title, color = colorList[int(title)-1])
            plt.plot([anomDetectIteration,anomDetectIteration], [0,maxVal], color = colorList[int(title)-1])
            plt.text(anomDetectIteration, maxVal, title)
            plt.legend()
            return anomDetectIteration, anomFalseDetectIteration, lossThresh
        else:
            plt.plot(timeData, loss, label = title, color = colorList[index])
            plt.legend()
            plt.plot(timeData, [lossThresh for i in range(len(loss))], label = title, color = colorList[index])
            print([anomDetectIteration])
            plt.plot([timeData[anomDetectIteration],timeData[anomDetectIteration]], [0,maxVal], color = colorList[index])
            plt.text(timeData[anomDetectIteration], maxVal, title)
            plt.legend()

    return timeData[anomDetectIteration], anomFalseDetectIteration, lossThresh  
