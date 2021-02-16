# -*- coding: utf-8 -*-
#"""
#Compiled on mon Feb 15 2021

#@author: Harry Bowman & Jack Byrne (compiler)
#"""
# Functions which produce visuals for BlueScope PowerBIDashborad

def waterFallPrep(specPdf):
	from datetime import datetime
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
	latestSpecsDf = bucketData(a, numBuckets = 16)
	return latestSpecsDf