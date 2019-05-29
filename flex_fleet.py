# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:32:56 2019

@author: Alex Palomino
"""

# Import Libraries
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import timeit
import time
import datetime


#%% Import Data

# Raw Data
filePath = 'data/Session-Details-Summary-20190404.csv';

# Import Data
data = pd.read_csv(filePath);

#dataHead = data.head(100);
#dataTypes = data.dtypes;

allColumns = list(data);

#%% Dataframe Preparation

def filterPrep(df, string, fltr):
    
    colNames = ['EVSE ID', 'Port Number', 'Station Name', 'Plug In Event Id', 'Start Date', 'End Date', 
            'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)', 'Energy (kWh)',
            'Ended By', 'Port Type', 'City', 'Latitude', 'Longitude', 'User ID', 'Driver Postal Code'];
            
    df = pd.DataFrame(df, index=np.arange(len(df)), columns=colNames)
    
    df['Start Date'] = pd.to_datetime(df['Start Date']);
    df['End Date'] = pd.to_datetime(df['End Date']);
    df['Total Duration (hh:mm:ss)'] = pd.to_timedelta(df['Total Duration (hh:mm:ss)']);
    df['Charging Time (hh:mm:ss)'] = pd.to_timedelta(df['Charging Time (hh:mm:ss)']);
    
    #filter by City
    if fltr:
        df = df[df['City'].str.contains(string)]
        print("Filter for: ", string)
    else:        
        print("No Filter")
           
    #clean data
    df = df.loc[df['Energy (kWh)'] > 0]
    df = df.loc[~pd.isnull(df['End Date'])]
    
    #update data types
    df['Duration (h)'] = df['Total Duration (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
    df['Duration (h)'] = df['Duration (h)'].apply(lambda x: round(x * 2) / 4) 
    df['Charging (h)'] = df['Charging Time (hh:mm:ss)'].apply(lambda x: x.seconds/3600)    
    df['Charging (h)'] = df['Charging (h)'].apply(lambda x: round(x * 2) / 4) 
    
    # Day of year 0 = Jan1 and day of year 365 = Dec31
    df['DayofYr'] = df['Start Date'].apply(lambda x: x.dayofyear) 
    # Monday is 0 and Sunday is 6
    df['DayofWk'] = df['Start Date'].apply(lambda x: x.weekday()) 
    df['Year'] = df['Start Date'].apply(lambda x: x.year) 
    df['StartHr'] = df['Start Date'].apply(lambda x: x.hour + x.minute/60) 
    df['StartHr'] = df['StartHr'].apply(lambda x: round(x * 4) / 4) 
    df['EndHr'] = df['End Date'].apply(lambda x: x.hour + x.minute/60) 
    df['EndHr'] = df['EndHr'].apply(lambda x: round(x * 4) / 4) 
    df['AvgPwr'] = df['Energy (kWh)']/df['Duration (h)']
    df['Date'] = df['Start Date'].apply(lambda x: str(x.year) + '-' + str(x.month) + '-' + str(x.day)) 
        
    df = df.loc[df['Duration (h)'] > 0]
    
    # Sort Dataframe
    df.sort_values(['Start Date'], inplace=True);
    df = df.reset_index(drop=True);

    # Assign Day Count    
    df['dayCount'] = 0;
    
    days = list(df['Start Date'].apply(lambda x: str(x.year) + '-' + str(x.month) + '-' + str(x.day)))
    daysSet = sorted(set(days), key=days.index)
    
    c = 0;
    for d in daysSet:
        
        dateTest = [df['Date'] == d]
        trueIdx = list(dateTest[0][dateTest[0]].index)
        df.at[trueIdx,'dayCount'] = c
        c += 1; 
        
    daysTot =  (df['Start Date'].iloc[len(df)-1] - df['Start Date'].iloc[0]).days+1
    
    return df, daysTot;


#%% Salt Lake City Sessions
    
dfSLC, daysTot = filterPrep(data, "Salt Lake City", True)

#%% Training and Testing

def testTrain(dfSLC, p):
    dfTrain = dfSLC.sample(int(p*len(dfSLC)))
    
    # Indices of Training Data
    idxTrain = list(dfTrain.index.values)
    # Indices of All Data
    idxdf = list(dfSLC.index.values)
    # Indices of Test Data
    idxTest = list(set(idxdf) - set(idxTrain))
    
    dfTest = dfSLC.iloc[idxTest]
    
    return dfTrain, dfTest

# Inputs (dfAll, percent Training Data)
dfTrain, dfTest = testTrain(dfSLC, 0.80)


