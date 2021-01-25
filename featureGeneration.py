def peaksByPeaks(specPdf):
    widthK = 20
    width = 10
    stepSize = 5
    promSF = 5
    spectrumLen = 512

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
    #Drop the rows of small magnitudes
    finalPdfNosmall = finalPdf.loc[finalPdf.magnitude > 0.2]

    finalDf = spark.createDataFrame(finalPdfNosmall)

    return finalDf