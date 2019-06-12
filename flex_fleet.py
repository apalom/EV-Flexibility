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

#%% Training and Testing for a Single Day

def testTrain(df, day, p):
    
    df = df.loc[df.DayofWk == day]
    df = df.reset_index(drop=True)
    
    dfTrain = df.sample(int(p*len(df)))
    
    # Indices of Training Data
    idxTrain = list(dfTrain.index.values)
    # Indices of All Data
    idxdf = list(df.index.values)
    # Indices of Test Data
    idxTest = list(set(idxdf) - set(idxTrain))
    
    dfTest = df.iloc[idxTest]
    
    return dfTrain, dfTest

# Inputs (dfAll, Day of Week [Mon = 0, Sat = 5] ,percent Training Data)
dfTrain, dfTest = testTrain(dfSLC, 0, 0.80)

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
# quants(df, weekday = True/False)
dfWkdy, quantData = quants(dfSLC, True)

#%% 

dfMonday = dfSLC.loc[dfSLC.DayofWk == 0]
dfMonday = dfMonday.reset_index(drop=True)

#%%

dim = 'Energy (kWh)'
stdDev2 = int(dfMonday[dim].quantile(q=0.977))


dfMon = dfMonday.loc[dfMonday[dim] < stdDev2]

# Sturge’s Rule for Bin Count
kBins = 1 + 3.22*np.log(len(dfMon))
# results approximately in 1 kWh wide bins for Energy

dfMon[dim].plot.hist(grid=True, bins=np.arange(0,25,1), 
                     density=True, rwidth=0.9, color='#607c8e')
                                    
mean, var = np.mean(dfMon[dim]), np.var(dfMon[dim])

#%%
import seaborn as sns

sns.set_style('darkgrid')
sns.distplot(dfMon[dim])

#%% Plot Fits

import scipy
from scipy import stats

sns.set_color_codes()

ax = sns.distplot(dfMon[dim], fit=stats.norm, kde=False,  
                  fit_kws={'color':'blue', 'label':'norm'})

ax = sns.distplot(dfMon[dim], fit=stats.gamma, hist=False, kde=False,  
                  fit_kws={'color':'green', 'label':'gamma'})

ax = sns.distplot(dfMon[dim], fit=stats.beta, hist=False, kde=False,  
                  fit_kws={'color':'red', 'label':'beta'})

ax = sns.distplot(dfMon[dim], fit=stats.skewnorm, hist=False, kde=False,  
                  fit_kws={'color':'grey', 'label':'skewnorm'})

ax.legend()

#%% Test Fits

x = dfMon[dim]
dists = ['norm','gamma','beta','skewnorm']

distribution = "beta"

for d in dists: 
    distr = getattr(stats, d)
    params = distr.fit(x)
    result = stats.kstest(x, d, args=params, N=1000)
    print(d, result)


#%%

ax = sns.distplot(dfMon[dim], rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 1, "label": "KDE"},
                   hist_kws={"histtype": "step", "linewidth": 1.5,
                             "alpha": 1, "color": "g"})

#%% Plot Connected Quants

quantData.plot()

plt.xlabel('Time (hr)')
plt.ylabel('Count')
plt.title('EV Sessions Started, SLC-Weekday')
plt.xlim(0,24)
plt.xticks(np.arange(0,26,2))

#%% Create a Sessions Plot

from matplotlib import pyplot as plt
plt.figure(figsize=(10,8))

i = 0;
dfSessions = pd.DataFrame(columns=['StartHr','Charging','Duration','Energy'])

for idx,row in dfSLC.sample(2000).iterrows():
    i += 1;
    dfSessions.at[i] = [row.StartHr, row['Charging (h)'], row['Duration (h)'], row['Energy (kWh)']]
    plt.plot((row.StartHr,row.EndHr),(row['Energy (kWh)'], row['Energy (kWh)']),
             linewidth=0.5, alpha=0.66)

plt.xlabel('Time (hr)')
plt.ylabel('Energy (kWh)')
plt.title('EV Sessions Energy, SLC-Weekday')
plt.xlim(0,24)
plt.xticks(np.arange(0,26,2))

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





