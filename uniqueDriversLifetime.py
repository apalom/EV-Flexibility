# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:18:37 2019

@author: Alex
"""

import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import timeit
import time
import datetime

#%% Import System Data

# Import Data
dataRaw = pd.read_excel('data/Session-Details-Summary-20200113.xlsx');
data = dataRaw;
dataHead = data.head(100);
dataTypes = data.dtypes;

allColumns = list(data);

#%% Unique Drivers per Port

colNames = ['Start Date', 'EVSE ID', 'Port Number', 'Port Type',
            'User ID', 'Driver Postal Code', 'DayofYr']
            
#firstDate = data.iloc[0]
dataUnique = pd.DataFrame(data, index=np.arange(len(data)), columns=colNames)

dataUnique['Date'] = dataUnique['Start Date'].apply(lambda x: x.date())
dataUnique['EVSE ID'] = dataUnique['EVSE ID'].map(str) + '-' + dataUnique['Port Number'].map(str)

dataUnique = dataUnique.drop(['Start Date', 'Port Number'], axis=1)

#%% Create List of Dates

import datetime

start = datetime.datetime.strptime("2013-01-01", "%Y-%m-%d")
end = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
#date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
date_generated =[x.date() for x in date_generated]
#date_generated =[x.date().strftime('%Y-%m-%d') for x in date_generated]

#%% Calculate Aggregate Lifetime Values

dfUnique = pd.DataFrame(np.zeros((len(date_generated),4)), columns=['Date','Drivers', 'DriversDay', 'Ports'])

for i in range(len(date_generated)): 
    
    d = date_generated[i]
    print(i,d)
        
    dfTemp = dataUnique.loc[dataUnique.Date <= d]        
    if len(dfTemp) > 0:
        dfUnique.iloc[i].Drivers = len(set(dfTemp['User ID'].values)) 
        dfUnique.iloc[i].Ports = len(set(dfTemp['EVSE ID'].values))   
        
    dfTempDay = dataUnique.loc[dataUnique.Date == d]
    dfUnique.iloc[i].DriversDay = len(set(dfTempDay['User ID'].values))
        
dfUnique.Date = date_generated    

#%% Lifetime Plots

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_data(data):
    
    x = data.Date;
    y_driversAgg = data.Drivers;
    y_driversDay = data.DriversDay;
    y_portsAgg = data.Ports;
    
    return x, y_driversAgg, y_driversDay, y_portsAgg 

x, y_driversAgg, y_driversDay, y_portsAgg = plot_data(dfUnique2016);

plt.style.use('default')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 12}
plt.rc('font', **font)

fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(x, y_driversAgg, c='k')
ax1.set_xlabel('Date')
yr_labels = ['2016-01', '2016-07', '2017-01', '2017-07', '2018-01', '2018-07', '2019-01', '2019-07', '2020-01']
ax1.set_xticklabels(yr_labels, rotation=45)
ax1.set_ylabel('Unique Drivers')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(x, y_portsAgg, '--', c='darkgrey')
ax2.set_ylabel('Ports Installed')

fig.legend(['Drivers','Ports'], loc='upper left', bbox_to_anchor=(0.15, 0.90))
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.show()

fig.savefig('results\lifetime2016.pdf')

#%% Date Filter

dfUnique2016 = dfUnique[(dfUnique.Date < datetime.date(2020, 1, 1))]
dfUnique2016 = dfUnique2016[(dfUnique2016.Date > datetime.date(2016, 1, 1))]
dfUnique2016 = dfUnique2016.reset_index(drop=True)

#%% Lifetime Plot 

plt.style.use('default')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 12}
plt.rc('font', **font)

fig = plt.subplots(figsize=(6,4))

plt.plot(x, y_driversDay, c='grey', lw=0.4, label='Drivers Per Day')
plt.plot(x, y_portsAgg, c='k', label='Ports')

plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.tight_layout()
plt.legend()
plt.show()

#fig.savefig('results\lifetime2016_perDay.pdf')

#%% Unique Drivers per Port

data.Date = pd.to_datetime(data.Date)

data = data[(data.Date < '2019-01-17 00:00:00')]
sub = 30;

df_ratio = []

for d in range(0, len(data) - sub, sub):
    subset = data.iloc[d:d+sub]
    avgDrivers = np.average(subset['Unique Drivers'])
    avgPort = np.average(subset['No. of Ports'])
    ratio = avgDrivers/avgPort;
    #print(data.Date.iloc[d], ratio)
    if ratio > 0 and ratio < np.inf:   
        df_ratio.append([data.Date.iloc[d], ratio])

df_ratio = pd.DataFrame(df_ratio, columns=['Date', 'Ratio'])

plt.bar(df_ratio.Ratio)
#plt.plot(data.Date, data['No. of Ports'], 'black')
#plt.plot(data.Date, data['Unique Drivers'], color='grey', alpha=0.5) 
#
#plt.legend()
