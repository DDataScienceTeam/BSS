# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:07:21 2020

@author: Harry Bowman
"""
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import glob
import matplotlib.pyplot as plt
import os,argparse
import time
from torch.utils.data import random_split
import time
from scipy.signal import butter,filtfilt
import pandas as pd

# from KPAE_funct import *
class kPeaksDataset(object):
    def __init__(self, data):
        self.features = []
        self.index = []
        for index in range(data.shape[0]):
            features = torch.Tensor(data[index,:])
            # print(features.shape)
            self.index.append(index)
            self.features.append(features)
            

    def __getitem__(self, idx):
        return self.features[idx]


    def __len__(self):
        return len(self.features)                      

#%% Autoencoder function
class FC_AE(nn.Module):
    def __init__(self, inSize = 16,midSize = 10, lowSize = 3):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(in_features=inSize, out_features=midSize),
                nn.ReLU(True),
                nn.Linear(in_features=midSize, out_features=lowSize),
                nn.ReLU(True)
            )
        self.decoder = nn.Sequential(     
                nn.Linear(in_features=lowSize, out_features=midSize),
                nn.ReLU(True),
                nn.Linear(in_features=midSize, out_features=inSize),
                )
    def forward(self,x):
        y = self.encoder(x)
        z = self.decoder(y)
        return y,z



def makeDataloader(data, colDrop = [], testSamples = 30, batchSize = 4):
    features = data.drop(columns = colDrop)
    mat = features.to_numpy()
    mag = mat[:, :10]
    freq = mat[:,10:]
    freqNorm = freq/np.max(freq)
    magNorm = mag/np.max(mag)
    featuresNorm = np.concatenate((magNorm, freqNorm),axis=1)
    
    #Make datasets and dataloaders
    datasetFull = kPeaksDataset(featuresNorm)
    l = len(datasetFull)
    torch.manual_seed(10)
    # print(l)
    datasetTrain, datasetVal = random_split(datasetFull, [l-testSamples, testSamples])
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=batchSize, shuffle=True)
    dataloaderVal = torch.utils.data.DataLoader(datasetVal, batch_size=1, shuffle=True)
    
    return dataloaderTrain, dataloaderVal, np.max(mag), np.max(freq)

def makeModelItems(lr = 1e-3, midSize = 16, smallSize = 10, inSize = 16):
    model = FC_AE(midSize = midSize, lowSize = smallSize, inSize = inSize)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion

def trainModel(dataloaderTrain, dataloaderVal, model, optimizer, criterion, numEpochs = 10, iteration = '1'):
    lossOverEpochs = []
    valLossOverEpochs = []
    for epoch in range(numEpochs):
        print('Epoch :',epoch)
        epochTime = time.time()
        epochLoss = 0
        for sample in dataloaderTrain:
            optimizer.zero_grad()
               
            # compute reconstructions
            _,output = model(sample)
            
            # compute training reconstruction loss
            train_loss = criterion(output, sample)
            
            # compute accumulated gradients
            train_loss.backward()
            epochLoss += train_loss.item()
            
            # perform parameter update based on current gradients
            optimizer.step()
        
        epochLoss = epochLoss/len(dataloaderTrain)
        lossOverEpochs.append(epochLoss)
        # print("epoch : {}/{}, recon loss = {:.8f}, time taken = {:.2f}".format(epoch + 1, numEpochs, epochLoss, (time.time() - epochTime)))
        
        #Validate results
        valLoss, predictArray, lossArray, inputArray = ValTest(dataloaderVal, model, criterion, finalEval = False)
        valLossOverEpochs.append(valLoss)
    plt.figure()
    plt.plot(valLossOverEpochs)
    plt.title('Iteration: ' + iteration + 'Lowest Loss = '+ str(min(valLossOverEpochs)))
    # plt.ylim([0,0.07])
    return model, valLossOverEpochs


def ValTest(dataloaderVal, model, criterion, finalEval = False, iteration = '1'):
    predictArray = np.array([])
    lossArray = np.array([])
    with torch.no_grad():
        valLoss = 0
        
        for j,sample in enumerate(dataloaderVal):                
            # compute reconstructions
            _,output = model(sample)
            
            # compute training reconstruction loss
            train_loss = criterion(output, sample)
            
            
            # calculate loss
            valLoss += train_loss.item()
            
            #Save items into arrays
            
            if j == 0:
                predictArray = output.detach().numpy()
                lossArray = np.abs((output - sample).detach().numpy())
                inputArray =  sample.detach().numpy()
            else:
                predictArray = np.concatenate((predictArray, output.detach().numpy()), axis = 0)
                lossArray = np.concatenate((lossArray,  np.abs((output-sample).detach().numpy())), axis = 0)
                inputArray = np.concatenate((inputArray, sample.detach().numpy()), axis = 0)
                
        valLoss = valLoss/len(dataloaderVal)
        # print("Loss for dataloader {} : {}".format(dlIndex, valLoss))
        # print(predictArray.shape)
        # print(lossArray.shape)
        
        if finalEval:        
            fig,ax = plt.subplots(3,1)
            for i in range(30):
                ax[0].plot(inputArray[i,:])
                ax[1].plot(predictArray[i,:])
                ax[2].plot(lossArray[i,:])
            ax[0].set_title('inputs')
            ax[1].set_title('predictions')
            plt.suptitle('Iteration '+iteration + ': Validation Input vs Samples')
    return valLoss, predictArray, lossArray, inputArray

def predictLoss(model, sample, criterion, magMax, freqMax):
    with torch.no_grad():
        mag = sample[:10]/magMax
        freq = sample[10:]/freqMax
        sampleNorm = np.concatenate((mag,freq))
        sampleTensor = torch.Tensor(sampleNorm)
        _,output = model(sampleTensor)
        loss = np.sum(np.abs(output.detach().numpy()- sampleTensor.detach().numpy()))
        # plt.figure()
        # plt.plot(output.detach().numpy())
        # plt.plot(sample.detach().numpy())
    return loss.item()

def butter_lowpass_filter(data, cutoff, fs, order, titleStr):
    normal_cutoff = cutoff / (0.5*fs)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    plt.figure()
    plt.plot(data)
    plt.plot(y)
    plt.title(titleStr)
    return y