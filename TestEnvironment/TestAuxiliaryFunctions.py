# -*- coding: utf-8 -*-
#"""
#Created on Monday 15 Feb 2021

#@author: Jack Byrne
#"""

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
    sm.create_table(Database,CopyName, table_def)




#-----------------------------------------------------------------------------------------------
# Takes in a Sparks Dataframe and takes some condition from another sparks data frame's column and toggles if data row is valid
# The output will be the Df with an updated Invalid column of 0 or 1
# Most parametres are pre-determined for the case of checking data against a 16000 battery charge, but can be changed
#------------------------------------------------------------------------------------------------
def CompInvalid(Df, ComparisonDf, deviceCols = 'deviceID', timestampCols = 'timestamp', ComparisonCols = 'battery', scaler = 1/1000, NegThreshold = True, Threshold = 16):
    Pdf = Df.to_Pandas()
    ComparisonPdf = ComparisonDf.to_Pandas()
    ## first creating helper columns in both Pdfs
    Pdf['date'] = pd.to_datetime(Pdf[timestampCols], format="%Y-%m-%dT%H:%M:%S.%fZ").dt.date # only concerned with date not timestamp
    if 'Invalid' not in Pdf.columns:
        Pdf['Invalid']= 0
    ComparisonPdf['date'] = pd.to_datetime(ComparisonPdf[timestampCols], format="%Y-%m-%dT%H:%M:%S.%fZ").dt.date 
    if 'Invalid' not in ComparisonPdf.columns:
        ComparisonPdf['Invalid']= 0 
    ## Average on dates to elimate issue of same date crossing the threshold 
    meandat = ComparisonPdf.groupby(['deviceID','date'], as_index=False)['battery'].mean().reset_index()
    ## Merge Pdfs for final analysis
    Pdf = Pdf.merge(meandat, on=[deviceCols, 'date'], how='left', suffixes=(None, '_meta'))
    ## Uses logic of abs() and NegThreshold 0 or 1 to combine both < and > in one line
    Pdf.loc[(((Pdf['battery']*scaler < Threshold)+int(NegThreshold)-1).abs()).astype(bool),'Invalid'] = 1
    Pdf = Pdf.drop(['date', 'index'], axis=1)
    return spark.createDataFrame(Pdf)

#-----------------------------------------------------------------------------------------------
# Takes in a Sparks Dataframe and takes some datetime.date condition oggles if data row is valid
# The output will be the Df with an updated Invalid column of 0 or 1
# Most parametres are pre-determined for the case of checking data against a 16000 battery charge, but can be changed
#------------------------------------------------------------------------------------------------
def TimeInvalid(Df, date, deviceCols = 'deviceID', timestampCols = 'timestamp', before=True):
    Pdf = Df.toPandas()
    Pdf['date'] = pd.to_datetime(Pdf[timestampCols], format="%Y-%m-%dT%H:%M:%S.%fZ").dt.date # only concerned with date not timestamp
    if 'Invalid' not in Pdf.columns:
        Pdf['Invalid']= 0
    
    if before==True:
        Pdf.loc[(Pdf['date']<date),'Invalid'] = 1
    elif before==False:
        Pdf.loc[(Pdf['date']>date),'Invalid'] = 1
    else:
        return print('before has an error')
    
    Pdf = Pdf.drop(['date'], axis=1)
    Output = spark.createDataFrame(Pdf)
    return Output   