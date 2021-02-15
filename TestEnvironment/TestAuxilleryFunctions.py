# -*- coding: utf-8 -*-
"""
Created on Monday 15 Feb 2021

@author: Jack Byrne
"""

from neuro_python.neuro_compute import spark_manager as spm
import neuro_python.neuro_compute.notebook_manager as notebook_manager
from neuro_python.neuro_data import schema_manager as sm
from neuro_python.neuro_data import sql_query as sq
from neuro_python.neuro_data import sql_commands as sc
import re
import plotly.plotly as py
import plotly.graph_objs as pgo
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import ipywidgets as widgets
import datetime as dt
import math
import time

#------------------------------------------------------------------
# Function to find key against its values. 
# In this context used to convert the datatypes into neuroverse friendly
#-------------------------------------------------------------------

def Keysearch(values, searchFor):
    for k in values:
        for v in values[k]:
            if searchFor == v:
                return k
    return None

#------------------------------------------------------------------
# Function to take a neuroverse table input and output a copy for either refresh or test duplicate purposes. 
#-------------------------------------------------------------------
# However this new table will need to be filled
# This should be done using a %%spark_export_table magic cell command.# SourceTable & Database must be string inputs, ST_Pdf must be a Pandas copy of the imported Source Table, 
# which can be done with %spark_pandas -df ST -o ST_Pdf magic cell command.
# By default this table will have the same name.

# Recommended to proceed this function with:
# %%spark_import_table 
# import_table('X', 'BluescopeReporting', 'Test_mlScore')

#%spark_pandas -df X -o X_pd

# then after run, fill data with:
# %%spark_export_table 
# export_table('X', 'BluescopeReporting', 'junk_test')

def TableCopy(SourceTable, Database, ST_Pdf, Delete = False, Test = False):
    # checking input variable types
    if not isinstance(SourceTable, str) or not isinstance(Database, str):
        return print("Names must be strings")
    
    # Strips out the column names and data types from the pandas copy
    rawList = ST_Pdf.dtypes
    Columns = []
    for i in range(len(rawList)):
#         print(rawList.index[i])
#         print(rawList[i])
        Columns.append([rawList.index[i],rawList[i].name])
	#print(List)

    
    if not isinstance(Columns, list):
        return print("invalid Columns entry: not list")
    if not list or tuple in Columns:
        return print("invalid Columns entry: list contains invalid elements")
    if tuple in Columns:
        for i in range(len(Columns)):
            if len(Columns[i]) != 2:
                return print("invalid Columns entry: too many entries for one Column")
            if Columns[i] is tuple:
                Columns[i] = list(Columns[i])
                print(Columns[i])

    
    # deleting pre-existing table
    if Delete == True: 
        try: sm.delete_processed_table(Database, SourceTable)
        except:
            return print("Table to be deleted does not exist or can not be deleted")
    else:
        pass
    
    # Translating standard data types into acceptable Neuroverse Types
    TypeDict = {'String(30)': ['string','O','object','str','String(30)'],
                'Int': ['int','Int','Integer','int64'],
                'Double': ['double','Double','float64','float','complex'],
                'ByteArray': ['ByteArray','list'],
                'VarBinary': ['VarBinary','binary']
               } 
    
    for i in range(len(Columns)):
        Columns[i][0] = Columns[i][0].replace(" ", "") # spaces through a disasterous Neuroverse error 
        Columns[i][1] = Columns[i][1].replace(" ", "")
        #print(Columns[i][0])
        #print(Columns[i][1])
        NeuroType = Keysearch(TypeDict, Columns[i][1])
        #print(NeuroType)
        if NeuroType == None:
            return print("Could not convert data type to Neuroverse friendly")
        else:
            Columns[i][1] = NeuroType
        if list in Columns:
            for i in range(len(Columns)):
                Columns[i] = tuple(Columns[i])
    
    # Changing Columns into table ready format
    cols = []
    for i in range(len(Columns)):
        cols.append(sm.column_definition(Columns[i][0],Columns[i][1]))
    #print(cols)
    #Changng prefix if toggled
    if Test == True:
        CopyName = 'Test_'+SourceTable
    else:
        CopyName = SourceTable
    table_def=sm.table_definition(cols, 'Processed', file_type='delta')
    sm.create_table('BluescopeReporting','junk_test', table_def)


#------------------------------------------------------------------
# Function to output the simpler parts (temperature and velocity) on the dashboard
# Summary Page Donuts. 
#-------------------------------------------------------------------

def simpleBlowerTests(startTime, endTime, vThreshDf, metaDf, durationStr):
# print(now,rounded, roundedDayBefore, rounded4HourBefore)
    velThresh = vThreshDf.toPandas()
    metaFilt = metaDf.filter(metaDf['timestamp'] > startTime)
#     print(metaFilt)
    metaFilt2 = metaFilt.filter(metaFilt['timestamp']<endTime)
    meta = metaFilt2.toPandas()
    hour = pd.to_datetime(meta['timestamp'])
    hourList = []
    for i in hour:
        hourList.append(i.hour)
    hourNp = np.array(hourList)
    meta['hour'] = hourNp

    #Groupby, loop and find anomalies in temperature

    anomScoreListTemp = [0,0,0,0,0,0]
    anomScoreListVel = [0,0,0,0,0,0]
    tempWarningVal  =7
    tempBadVal = 15
    if durationStr == 'week':
        velWarningVal = 5 #num of points greater than for warning/bad
        velBadVal = 18
        tempWarningThresh = 20
        tempBadThresh = 35
    else:
        velWarningVal = 0 #num of points greater than for warning/bad
        velBadVal = 3
        tempWarningThresh = 1
        tempBadThresh = 3
        
        
    for hour, group in meta.groupby('hour'):
        totalMean = np.mean(group['temperature'].values)
    #     print(totalMean)
        for j,(deviceID, smallGroup) in enumerate(group.groupby('deviceID')):
            for i in range(smallGroup.shape[0]):
                if smallGroup.iloc[i].temperature > totalMean + tempBadVal:
                    anomScoreListTemp[j] = anomScoreListTemp[j] + 3
                    break
                elif smallGroup.iloc[i].temperature > totalMean + tempWarningVal:
                    anomScoreListTemp[j] = anomScoreListTemp[j] + 1
            
            if anomScoreListTemp[j] > tempBadThresh:
                anomScoreListTemp[j] = 2
            elif anomScoreListTemp[j] > tempWarningThresh:
                anomScoreListTemp[j] = 1
            else:
                anomScoreListTemp[j] = 0

    #find instances where the velocity exceeds the threshold
    for j, (deviceID, group) in enumerate(meta.groupby('deviceID')):
        vels = []
        for hour, hourGroup in group.groupby('hour'):
            vels.append(np.mean(hourGroup['Velocity'].values))
        vels = np.array(vels)
        thresh = velThresh.loc[velThresh['deviceID'] == deviceID]['velThresh'].values[0]
        #print(deviceID, (vels[vels>thresh]))
        if (len(vels[vels>thresh]))>velBadVal:
            anomScoreListVel[j] = 2
        elif (len(vels[vels>thresh]))>velWarningVal:
            anomScoreListVel[j] = 1
        else:
            anomScoreListVel[j] = 0

#     print(anomScoreListVel, anomScoreListTemp)



    ###########################
    #GET INTO DONUT FORM
    ##########################
    donutPdf = pd.DataFrame([], columns = ['deviceID', 'timeRecord','durationStr' ,'tempGood', 'tempWarning', 'tempBad','velGood', 'velWarning', 'velBad', 'velCard', 'tempCard'])
    blowerList = ['Blower 1', 'Blower 2', 'Blower 3', 'Blower 4', 'Blower 5', 'Blower 6']
#     timeRecord already made

    for i, (deviceID, group) in enumerate(meta.groupby('deviceID')):
        
        descript = np.array([blowerList[i], endTime, durationStr])
        if anomScoreListVel[i] == 2:
            velArr = np.array([0,0,1])
        elif anomScoreListVel[i] == 1:
            velArr = np.array([0,1,0])
        else:
            velArr = np.array([1,0,0])
        if anomScoreListTemp[i] == 2:
            tempArr = np.array([0,0,1])
        elif anomScoreListTemp[i] == 1:
            tempArr = np.array([0,1,0])
        else:
            tempArr = np.array([1,0,0])

        cardVals = np.array([round(group['Velocity'].iloc[0], 2), (str(math.floor(group['temperature'].iloc[0]))+'°')])

        x = np.append(descript, tempArr)
        y = np.append(x, velArr)
        row = np.append(y, cardVals)
        rowDf = pd.DataFrame([row], columns = ['deviceID', 'timeRecord', 'durationStr','tempGood', 'tempWarning', 'tempBad','velGood', 'velWarning', 'velBad', 'velCard', 'tempCard'])
        donutPdf = donutPdf.append(rowDf)
    return donutPdf


#------------------------------------------------------------------
# Function to output the the two device health circles (battery and ingestion)
# on the dashboard Summary Page.
#-------------------------------------------------------------------

def deviceTests(startTime, endTime, metaDf, specDf,durationStr):
    #This function is looking for various values and comparing them to levels to attain two values: batteryHealth and ingestionHealth
    #########################################################
    #Filter and prep data
    #########################################################
    #For spectrum 
    specFilt = specDf.filter(specDf['timestamp'] > startTime)
    specFilt2 = specFilt.filter(specFilt['timestamp']<endTime)
    specPdf = specFilt2.toPandas()
    #For Devices, to ensure all are processed in loop
    deviceList = ['Blower 1', 'Blower 2', 'Blower 3', 'Blower 4', 'Blower 5', 'Blower 6']
    numBlowers = 6
#     deviceList = ['Blower '+str(i+1) for i in range(numBlowers)]
    #For meta
    
    metaFilt = metaDf.filter(metaDf['timestamp'] > (endTime - td(days = 1)))
    metaFilt2 = metaFilt.filter(metaFilt['timestamp']<(endTime + td(days = 1)))
    metaPdf = metaFilt2.toPandas()
    
    #Verifying the metaDf filtering
#     print(metaDf)
#     print((metaFilt.count()))
#     print((metaFilt2.count()))
    
    #########################################################
    #Set param based on duration of analysis
    #########################################################
    if durationStr == 'week':
        ingestWarningThresh = 12*7
        ingestBadThresh = 4*7
        battThreshMid = 18
        battThreshLow  =16
    else:
        ingestWarningThresh = 12
        ingestBadThresh = 4
        battThreshMid = 18
        battThreshLow  =16

        
    anomScoreListIngest = []
    anomScoreListBatt = []
    lowerBound = -0.1
    battList = []
    ingestList = []
    #########################################################
    #DATA INGESTION
    #########################################################
    for deviceID in deviceList: # For each device
        group_spec = specPdf.loc[specPdf.deviceID == deviceID]
        group_meta = metaPdf.loc[metaPdf.deviceID == deviceID]
        # Limits dataframe to data
        group_spec = group_spec.drop(['deviceID', 'timestamp'], axis=1)  
        group_meta = group_meta.drop(['deviceID', 'timestamp'], axis=1)  
        
        sampleMeasure = 0
        velocityMeasure = 0
        specMeasure = 0
        
        #############################
        #Num Samples
        #############################
        numSamples = group_spec.shape[0]
        ingestList.append(numSamples)
        if numSamples < ingestBadThresh:
            sampleMeasure = 2
        elif numSamples < ingestWarningThresh:
            sampleMeasure=1
        else:
            sampleMeasure=0
            
        ###############################
        #Velocity
        ###############################
        vels = group_meta['Velocity'].values
        if (vels<lowerBound).any(): 
            velocityMeasure = 2
        else:
            velocityMeasure = 0
            
        ###############################
        #Spectrum
        ###############################
        if (group_spec.values<lowerBound).any():
            specMeasure = 2
        else:
            specMeasure = 0
            
        decisionList = np.array([specMeasure, velocityMeasure, sampleMeasure])
        anomScoreListIngest.append(np.max(decisionList))
#         print(group_meta.values)
        batt = np.mean(group_meta.battery.values)/1000
        battList.append(batt)
        if batt < battThreshMid:
            if batt < battThreshLow:
                anomScoreListBatt.append(2)
            else:
                anomScoreListBatt.append(1)
        else:
            anomScoreListBatt.append(0)
               


    ##########################
    deviceDonutPdf = pd.DataFrame([], columns = ['deviceID', 'timeRecord', 'durationStr','battGood', 'battWarning', 'battBad','ingestGood', 'ingestWarning', 'ingestBad'])
    blowerList = ['Blower 1', 'Blower 2', 'Blower 3', 'Blower 4', 'Blower 5', 'Blower 6']
    #timeRecord already made

    for i in range(6):
        descript = np.array([blowerList[i], endTime, durationStr])
        if anomScoreListBatt[i] == 2:
            battArr = np.array([0,0,1])
        elif anomScoreListBatt[i] == 1:
            battArr = np.array([0,1,0])
        else:
            battArr = np.array([1,0,0])
        if anomScoreListIngest[i] == 2:
            ingestArr = np.array([0,0,1])
        elif anomScoreListIngest[i] == 1:
            ingestArr = np.array([0,1,0])
        else:
            ingestArr = np.array([1,0,0])


        x = np.append(descript, battArr)
        row = np.append(x, ingestArr)
        rowDf = pd.DataFrame([row], columns = ['deviceID', 'timeRecord', 'durationStr', 'battGood', 'battWarning', 'battBad','ingestGood', 'ingestWarning', 'ingestBad'])
        deviceDonutPdf = deviceDonutPdf.append(rowDf)
    #Add in the batt levels and data level
    # print([(str(round(i,1))+'V') for i in battList])
    # print(len(battList))
    # print(len(ingestList))
    deviceDonutPdf['BattVals'] = [(str(round(i,1))+'V') for i in battList]
    deviceDonutPdf['DataLevels'] = ingestList #ITS A COUNT
    return deviceDonutPdf