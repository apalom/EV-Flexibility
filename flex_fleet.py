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
    yr = 2018
    df = df.loc[(df['Start Date'] > datetime.date(yr,1,1)) & (df['Start Date'] < datetime.date(yr+1,1,1))]
    
    
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
    df['StartHr'] = df['StartHr'].apply(lambda x: round(x)) 
    #df['StartHr'] = df['StartHr'].apply(lambda x: round(x * 4) / 4) 
    df['EndHr'] = df['End Date'].apply(lambda x: x.hour + x.minute/60) 
    #df['EndHr'] = df['EndHr'].apply(lambda x: round(x * 4) / 4) 
    df['EndHr'] = df['EndHr'].apply(lambda x: round(x)) 
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


#% Salt Lake City Sessions
    
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



#%% Calculate Mean, 1st and 2nd Standard Deviation of Connected Vehicles

def quants(df, weekday):

    allDays = list(set(df.dayCount))
    
    if weekday:
        df = df[df.DayofWk < 5]
    else:
        df = df[df.DayofWk >= 5]
    
    daysIn = list(set(df.dayCount))
    daysIn.sort()
    
    dfDays = pd.DataFrame(np.zeros((24,len(set(df.dayCount)))), 
                        index= np.arange(0,24,1), columns=daysIn)
    
    for d in df.dayCount:
    
        dfDay = df[df.dayCount == d]
        cnct = dfDay.StartHr.value_counts()
        cnct = cnct.sort_index()
        
        dfDays.loc[:,d] = dfDay.StartHr.value_counts()
        dfDays.loc[:,d] = np.nan_to_num(dfDays.loc[:,d])
    
    quants = pd.DataFrame(np.zeros((24,5)), 
                        index= np.arange(0,24,1), 
                        columns=['-2_sigma','-1_sigma','mu','+1_sigma','+2_sigma'])
    
    quants['-2_sigma'] = dfDays.quantile(q=0.023, axis=1)
    quants['-1_sigma'] = dfDays.quantile(q=0.159, axis=1)
    quants['mu'] = dfDays.quantile(q=0.50, axis=1)
    quants['+1_sigma'] = dfDays.quantile(q=0.841, axis=1)
    quants['+2_sigma'] = dfDays.quantile(q=0.977, axis=1)

    return dfDays, quants

dfWkdy, quants = quants(dfSLC, True)

quants.plot()

#%% Poisson Distribution Fit

from scipy.stats import poisson

N = 21;

poissMatch = np.zeros((N,3))

for n in range(1, N):
    
    mean, var, skew, kurt = poisson.stats(n, moments='mvsk')
    print(mean, var)
    poissMatch[n,0] = mean
    poissMatch[n,1] = mean + var
    poissMatch[n,2] = mean - var
    
#%% Poisson Distribution Fit
    
trials = 20000
N = 21


poissMatch = pd.DataFrame(np.zeros((3*N,3)), columns=['mu','val','cat'])
poissTrial = pd.DataFrame(np.zeros((trials,2)), columns=['mu','val'])

for n in range(0,N):
    
    poisRdm = np.random.poisson(n, trials)
    poissTrial.iloc[n:trials+(n)] = [np.array(trials*([n]-1)), poisRdm]
    
    mu = np.mean(poisRdm)
    poissMatch.at[n] = [mu, mu, 'mu']
    
    var = np.var(poisRdm)  
    poissMatch.at[n+N] = [mu, mu-var, '-var']
    
    var = np.var(poisRdm)
    poissMatch.at[n+2*N] = [mu, mu+var, '+var']
    

#%% Plot Poisson Distribution

import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.0})

colors = ["black", "grey", "grey", ]

# Show the results of a linear regression within each dataset
lm = sns.lmplot(x="mu", y="val", hue="cat", data=poissMatch, scatter=True,
           palette=sns.xkcd_palette(colors), scatter_kws={"s": 25, "alpha": 1})

lm.axes[0,0].set_xlim(0,)

#%%

evCount = dfWkdy.values.flatten()

#%%

xD = np.mean(dfWkdy,axis=1)
yD = np.var(dfWkdy,axis=1)

plt.scatter(xD, yD)
plt.xlim(0,16)
plt.xlim(0,16)

#%%

time = np.zeros((21,1))
time = np.arange(0,21)

idxMat = np.full((20000,21), time)

#idx1 = np.repeat(time, N, axis=1)

plt.scatter(x=idxMat,y=dfPoissTrial.values)

#%%

dfPoissTrial = pd.DataFrame(data=poissTrial)
sns.regplot(x=dfPoissTrial.index,y=dfPoissTrial.values,data=dfPoissTrial)





