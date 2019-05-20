# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:55:18 2019

@author: Alex
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

def filterPrep(df, string, contains):
    
    colNames = ['EVSE ID', 'Port Number', 'Station Name', 'Plug In Event Id', 'Start Date', 'End Date', 
            'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)', 'Energy (kWh)',
            'Ended By', 'Port Type', 'Latitude', 'Longitude', 'User ID', 'Driver Postal Code'];
            
    df = pd.DataFrame(df, index=np.arange(len(df)), columns=colNames)
    
    df['Start Date'] = pd.to_datetime(df['Start Date']);
    df['End Date'] = pd.to_datetime(df['End Date']);
    df['Total Duration (hh:mm:ss)'] = pd.to_timedelta(df['Total Duration (hh:mm:ss)']);
    df['Charging Time (hh:mm:ss)'] = pd.to_timedelta(df['Charging Time (hh:mm:ss)']);
    
    #filter by string name
#    if contains:
#        df = df[df['Station Name'].str.contains(string)]
#        print("Packsize Chargers")
#    else:
#        df = df[~df['Station Name'].str.contains(string)]
#        print("All Non-Packsize Chargers")
           
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

#dfPack, daysTot = filterPrep(data, "PACKSIZE", True)
#dfNotPack, daysTot = filterPrep(data, "PACKSIZE", False)
dfAll, daysTot = filterPrep(data, "PACKSIZE", False)

#%% Create Multi-Index Dictionary

def dfMulti(df, idxList):
    
    dfMulti = df.set_index(idxList)
    dfMulti = dfMulti.sort_index()

    dfDay = {}
    
    for index, temp_df in dfMulti.groupby(level=[0,1,2]):
        dfDay[index] = temp_df;
        
    return dfDay

idxSelect = ['DayofWk', 'dayCount', 'StartHr'];
#dfPackMulti = dfMulti(dfPacksize, idxSelect)

#%% Create Multi-Index DF Day 

def dfMulti(df, idxList):
    
    dfMulti = df.set_index(idxList)
    dfMulti = dfMulti.sort_index()

    return dfMulti

idxSelect = ['DayofWk', 'dayCount', 'StartHr'];
dfMulti = dfMulti(dfNotPack, idxSelect)

#%% Calculate Random Variable from Multi-Index DataFrame

def calcRVmulti(df, daysTot):
    
    # Create Multi-Index DF
    idxList = ['DayofWk', 'dayCount', 'StartHr'];
    df = df.set_index(idxList)
    df = df.sort_index()
    
    # Initialize Random Variable Dicitionary
    rv_Dict = {};
    avg_Dict = {};
    
    # Define Bin for RV Histogram    
    binHr = np.arange(0,24.0,0.25);
    binCar = np.arange(0,11,1);
    binKWH = np.arange(0,66,6);
    binDur = np.arange(0.50,6.00,0.50);
    binSprw = np.arange(0.1,1.2,0.1);       
    
    for dayOfWk in range(7):
                
        #daysTot = (df['Start Date'].iloc[len(df)-1] - df['Start Date'].iloc[0]).days+1
        
        cnctdPerDay = np.zeros((len(binHr),daysTot));
        energyPerDay = np.zeros((len(binHr),daysTot));
        durationPerDay = np.zeros((len(binHr),daysTot));
        sparrowPerDay = np.zeros((len(binHr),daysTot));
        
        rv_Connected = np.zeros((len(binHr),len(binCar)-1));
        rv_Energy = np.zeros((len(binHr),len(binKWH)-1));
        rv_Duration = np.zeros((len(binHr),len(binDur)-1));
        rv_Sparrow = np.zeros((len(binHr),len(binSprw)-1));
        
        # Define Bin for RV Histogram    
        binHr = np.arange(0,24.0,0.25);
        binCar = np.arange(0,11,1);
        binKWH = np.arange(0,66,6);
        binDur = np.arange(0.50,6.00,0.50);
        binSprw = np.arange(0.1,1.2,0.1);
        
        #dfDay = df.iloc[df.index.get_level_values('DayofWk') == dayOfWk]
        dfDay = df.loc[dayOfWk]
        
        for idx in dfDay.index:
        
            dayNum, hr = idx[0], idx[1]
            
            # 15 min row
            if hr == 24:
                r = 0;
            else:   
                r = int(4*hr);
                
            print("--- Index: ", dayOfWk, idx)
                    
            #dfTemp = dfDay[idx]
            #dfTemp = pd.DataFrame(dfDay.xs((idx)))
            dfTemp = dfDay.xs((idx))
                                            
            cnctdPerDay[r,dayNum] = len(dfTemp);
            energyPerDay[r,dayNum] = np.sum(dfTemp['Energy (kWh)'])#.values);
            durationPerDay[r,dayNum] = np.sum(dfTemp['Duration (h)'])#.values);
        
            # Condition set to avoid division by zero
            if np.sum(dfTemp['Duration (h)']) > 0.0: #.values) > 0.0:
                sparrowPerDay[r,dayNum] = np.sum(dfTemp['Charging (h)'])/np.sum(dfTemp['Duration (h)'])
                #sparrowPerDay[r,dayNum] = np.sum(dfTemp['Charging (h)'].values)/np.sum(dfTemp['Duration (h)'].values)
        
            # Histogram
            n_cnctd = np.histogram(cnctdPerDay[r,:], bins=binCar, density=True);
            n_energy = np.histogram(energyPerDay[r,:], bins=binKWH, density=True);
            n_duration = np.histogram(durationPerDay[r,:], bins=binDur, density=True);
            n_sparrow = np.histogram(sparrowPerDay[r,:], bins=binSprw, density=True);
            
            rv_Connected[r,:] = n_cnctd[0];
            rv_Energy[r,:] = 6*n_energy[0];
            rv_Duration[r,:] = 0.5*n_duration[0];
            rv_Sparrow[r,:] = 0.1*n_sparrow[0];
        
        averages = {'avg_Connected': np.mean(cnctdPerDay, axis=1), 
                    'avg_Energy': np.mean(energyPerDay, axis=1),
                    'avg_Duration': np.mean(durationPerDay, axis=1), 
                    'avg_Sparrow': np.mean(sparrowPerDay, axis=1) 
                    }
        avg_Dict[dayOfWk] = averages;
        
        dicts = {'rv_Connected': rv_Connected, 'rv_Energy': rv_Energy, 'rv_Duration': rv_Duration, 'rv_Sparrow': rv_Sparrow }
        rv_Dict[dayOfWk] = dicts

#    rv_Dict[dayOfWk]['rv_Connected'][np.isnan(rv_Dict[dayNum]['rv_Connected'])] = 0
#    rv_Dict[dayOfWk]['rv_Energy'][np.isnan(rv_Dict[dayNum]['rv_Energy'])] = 0
#    rv_Dict[dayOfWk]['rv_Duration'][np.isnan(rv_Dict[dayNum]['rv_Duration'])] = 0
#    rv_Dict[dayOfWk]['rv_Sparrow'][np.isnan(rv_Dict[dayNum]['rv_Sparrow'])] = 0

    return rv_Dict, avg_Dict

flexParams, avgParams = calcRVmulti(dfAll, daysTot)

outputFlex(flexParams, "All-Flexibility")

#%% Plot Averages 

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Plot Style Definition
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.color'] = 'k'
matplotlib.rcParams['font.family'] = 'serif'


day = 1
t = np.arange(0,96)

# Define Sub-Figures
fig, ax = sns.plt.subplots(len(avgParams[day]), 1, figsize=(10,8))
fig.subplots_adjust(hspace=0.6)#, wspace=0.4)
plt.xlabel("Time")

for a, key in zip(ax,avgParams[day].keys() ):
    y = avgParams[day][key]
    n = len(y)
    x = np.linspace(1,n,n)
    a.plot(x,y)    
    
    # labels/titles and such here
    a.set(ylim=(0, 0.02))
    a.set(xticks = np.arange(0,96,4))
    a.set_xticklabels(np.arange(0,24,1))
    a.set_title(key)    

plt.show()



#%% Calculate Random Variables (Per Hr)

def calcRV(df):
    
    # Initialize Random Variable Dicitionary
    rv_Dict = {};

    # Monday is 0 and Sunday is 6
    daysOfWeek = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']
    
    c = 0; 
    daysTot = (df['Start Date'].iloc[len(df)-1] - df['Start Date'].iloc[0]).days+1
    
    # Define Bin for RV Histogram    
    binHr = np.arange(0,24.0,0.25);
    binCar = np.arange(0,11,1);
    binKWH = np.arange(0,66,6);
    binDur = np.arange(0.50,6.00,0.50);
    binSprw = np.arange(0.1,1.2,0.1);
    
    dates = list(set(df['Date']));

    for daySlct in range(len(daysOfWeek)):
        
        df = df.loc[df['DayofWk'] == daySlct]
        
        cnctdPerDay = np.zeros((len(binHr),daysTot));
        energyPerDay = np.zeros((len(binHr),daysTot));
        durationPerDay = np.zeros((len(binHr),daysTot));
        sparrowPerDay = np.zeros((len(binHr),daysTot));
        
        rv_Connected = np.zeros((len(binHr),len(binCar)-1));
        rv_Energy = np.zeros((len(binHr),len(binKWH)-1));
        rv_Duration = np.zeros((len(binHr),len(binDur)-1));
        rv_Sparrow = np.zeros((len(binHr),len(binSprw)-1));
        
        c = 0;
        
        for d in dates:
        
            dfTemp = df.loc[df['Date'] == d]
            print('\n Day:', c)
            
            r = 0;
            
            for hr in binHr:
            
                dfTemp1 = dfTemp.loc[dfTemp['StartHr'] == hr]
                cnctd = len(dfTemp1)
        
                cnctdPerDay[r,c] = cnctd;
                energyPerDay[r,c] = np.sum(dfTemp1['Energy (kWh)'].values)
                durationPerDay[r,c] = np.sum(dfTemp1['Duration (h)'].values)
        
                if np.sum(dfTemp1['Duration (h)'].values) > 0.0:
                    sparrowPerDay[r,c] = np.sum(dfTemp1['Charging (h)'].values)/np.sum(dfTemp1['Duration (h)'].values)
    
                # Histogram
                n_cnctd = np.histogram(cnctdPerDay[r,:], bins=binCar, density=True);
                n_energy = np.histogram(energyPerDay[r,:], bins=binKWH, density=True);
                n_duration = np.histogram(durationPerDay[r,:], bins=binDur, density=True);
                n_sparrow = np.histogram(sparrowPerDay[r,:], bins=binSprw, density=True);
                
                rv_Connected[r,:] = n_cnctd[0];
                rv_Energy[r,:] = 6*n_energy[0];
                rv_Duration[r,:] = 0.5*n_duration[0];
                rv_Sparrow[r,:] = 0.1*n_sparrow[0];
                
                r += 1;   
            c += 1;
        
        dicts = {'rv_Connected': rv_Connected, 'rv_Energy': rv_Energy, 'rv_Duration': rv_Duration, 'rv_Sparrow': rv_Sparrow }
        rv_Dict[daySlct] = dicts

        rv_Dict[daySlct]['rv_Connected'][np.isnan(flexParams[daySlct]['rv_Connected'])] = 0
        rv_Dict[daySlct]['rv_Energy'][np.isnan(flexParams[daySlct]['rv_Energy'])] = 0
        rv_Dict[daySlct]['rv_Duration'][np.isnan(flexParams[daySlct]['rv_Duration'])] = 0
        rv_Dict[daySlct]['rv_Sparrow'][np.isnan(flexParams[daySlct]['rv_Sparrow'])] = 0
    
    return rv_Dict

flexParams = calcRV(dfPacksize)
outputFlex(flexParams, "PackSize-Flexibility-r0")

#%% Output Random Variable to CSV

# What does covariance give us? https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html
    # Covariance indicates the level to which two variables vary together.
    # The covariance matrix element C_{ij} is the covariance of x_i and x_j. The element C_{ii} is the variance of x_i.
    # Covariance Matrix takes random variables (in rows) and observations (in columns) and calculates the covariance
    # between (i,j) pairs of random variable reailizations.

def outputFlex(flexParams, outputPath):

    for d in range(7):
    
        fileName = 'exports\\' + outputPath + '\\' + str(d) + '-RandomVariable.xlsx'
                
        writer = pd.ExcelWriter(fileName, engine='xlsxwriter')
           
        pd_C = pd.DataFrame(flexParams[d]['rv_Connected'])    
        pd_D = pd.DataFrame(flexParams[d]['rv_Duration'])    
        pd_E = pd.DataFrame(flexParams[d]['rv_Energy'])    
        pd_S = pd.DataFrame(flexParams[d]['rv_Sparrow'])
        
        pd_C.to_excel(writer, sheet_name='rv_Connected')    
        pd_D.to_excel(writer, sheet_name='rv_Duration')    
        pd_E.to_excel(writer, sheet_name='rv_Energy')    
        pd_S.to_excel(writer, sheet_name='rv_Sparrow')
        
    writer.save()
        
#outputFlex(flexParams, "PackSize-Flexibility")


#%% Output Covariance for Each TimeStep to CSV
# Return covariance calculations on flexibility random variable parameters
    
def outputCovAvg(avgParams, outputPath):

    for d in range(7):
    
        fileName = 'exports\\' + outputPath + '\\' + str(d) + '-Covariance.xlsx'
                
        writer = pd.ExcelWriter(fileName, engine='xlsxwriter')
        
        covTemp = np.vstack((avgParams[d]['avg_Connected'], avgParams[d]['avg_Energy'], 
                             avgParams[d]['avg_Duration'], avgParams[d]['avg_Sparrow']));
    
        # Remove NaNs
        where_are_NaNs = np.isnan(covTemp)
        covTemp[where_are_NaNs] = 0
        
        labels = ['avg_Connected', 'avg_Energy', 'avg_Duration', 'avg_Sparrow']
        avg_Cov = pd.DataFrame(np.cov(covTemp), index=labels, columns=labels)
                                
        avg_Cov.to_excel(writer, sheet_name='avg_Cov')
 
    writer.save()

outputCovAvg(avgParams, "PackSize-Flexibility")


#%% Output Covariance for Each TimeStep to CSV
# Return covariance calculations on flexibility random variable parameters
    
def outputCov15(flexParams, outputPath):

    for d in range(7):
    
        fileName = 'exports\\' + outputPath + '\\' + str(d) + '-Covariance.xlsx'
                
        writer = pd.ExcelWriter(fileName, engine='xlsxwriter')
                
        pd_C_cov = pd.DataFrame(np.cov(flexParams[d]['rv_Connected']))
        pd_D_cov = pd.DataFrame(np.cov(flexParams[d]['rv_Duration']))
        pd_E_cov = pd.DataFrame(np.cov(flexParams[d]['rv_Energy']))
        pd_S_cov = pd.DataFrame(np.cov(flexParams[d]['rv_Sparrow']))
        
        pd_C_cov.to_excel(writer, sheet_name='cov_Connected')
        pd_D_cov.to_excel(writer, sheet_name='cov_Duration')
        pd_E_cov.to_excel(writer, sheet_name='cov_Energy')
        pd_S_cov.to_excel(writer, sheet_name='cov_Sparrow')
        
    writer.save()

outputCov15(flexParams, "PackSize-Flexibility")


#%% EVSE IDs

def getID(df):

    evse_ID = list(set(df['EVSE ID']))
    evseDict = {}
    
    for id in evse_ID:
        dfTemp = df.loc[df['EVSE ID'] == id]
        name = dfTemp['Station Name'].iloc[0]
        evseDict[id] = name
    
    return evseDict
        
dfPacksizeIDs = getID(dfPacksize)
