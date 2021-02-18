# -*- coding: utf-8 -*-
#"""
#Compiled on mon Feb 15 2021

#@author: Harry Bowman & Jack Byrne (compiler)
#"""
# Functions which produce visuals for BlueScope PowerBIDashborad


#------------------------------------------------------------------
# Function to output the waterfall spectrum diagram. 
#-------------------------------------------------------------------
def waterFallPrep(specPdf):
    blowerList = ['Blower '+str(i+1) for i in range(6)]
    b = specPdf
    dateObj = (pd.to_datetime(b['timestamp'])).dt
    day = dateObj.day
    month = dateObj.month
    year = dateObj.year
    dayVal = day + month*31 + (year-2020)*365
    b['dayVal'] = dayVal
    for blower in blowerList:
        c = b.loc[b['deviceID']==blower]
        for j,(dayVal,group) in enumerate(c.groupby('dayVal')):
            dataMean = group.mean(axis=0).values
            dataMean = np.concatenate((np.array([blower]), dataMean), axis=0)
            if blower == 'Blower 1' and j==0:
                dataFull = np.expand_dims(dataMean, axis=0)
            else:
                dataFull = np.concatenate((dataFull, np.expand_dims(dataMean, axis=0)), axis=0)
                
    #print(dataFull.shape)
    columns = ['deviceID'] + [str(i) for i in range(512)] + ['timestamp']
    a = pd.DataFrame(dataFull, columns = columns)
    a = a.iloc[:, 1:512].astype(float) # to solve bug of all data being string
    latestSpecsDf = bucketData(a, numBuckets = 16)
    return latestSpecsDf

#------------------------------------------------------------------
# Function to output table to power the 2 simpler segments of the Dashboard
# Summary Donuts
#-------------------------------------------------------------------
def simpleBlowerTests(startTime, endTime, vThreshDf, metaDf, durationStr):
# print(now,rounded, roundedDayBefore, rounded4HourBefore)
    velThresh = vThreshDf.toPandas()
    metaFilt = metaDf.filter(metaDf['timestamp'] > startTime)
    #print(metaFilt)
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
            
     #print(anomScoreListVel, anomScoreListTemp)



    ###########################
    #GET INTO DONUT FORM
    ##########################
    donutPdf = pd.DataFrame([], columns = ['deviceID', 'timeRecord','durationStr' ,'tempGood', 'tempWarning', 'tempBad','velGood', 'velWarning', 'velBad', 'velCard', 'tempCard'])
    blowerList = ['Blower 1', 'Blower 2', 'Blower 3', 'Blower 4', 'Blower 5', 'Blower 6']
     #timeRecord already made

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
    donutPdf['velCard'] = pd.to_numeric(donutPdf['velCard'], downcast='float')      
#     donutDf = spark.createDataFrame(donutPdf)
    return donutPdf


#------------------------------------------------------------------
# Function to output the two device health circles the Dashboard Summary
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

    #Add in the batt levels and data levels
   # print([(str(round(i,1))+'V') for i in battList])
    #print(len(battList))
    #print(len(ingestList))
    deviceDonutPdf['BattVals'] = [(str(round(i,1))+'V') for i in battList]
    deviceDonutPdf['DataLevels'] = ingestList #ITS A COUNT
    return deviceDonutPdf