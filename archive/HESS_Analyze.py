# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:48:36 2018

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

# Raw Data
path = 'exports\\data_XF1003-PackSize-2018-06-20to2018-12-10.csv'
#path = 'exports\\data_XF1001_Bus-2017-12-15to2018-01-15.csv'

# Import Data
dataRaw = pd.read_csv(path)
data = dataRaw
dataHead = data.head(100);

# Prepped Data
#path = 'exports\\outputFile01.csv'
#data = pd.read_csv(path)

#%% Prep Data

colNames = ['TIME', 'HOUR', 'DAY', 'WEEKDAY', 'KWH', 'KWHadded', 'SESSION', 'KVAH', 'KVA', 'KW', 'KVAR', 'PF','VRMSA', 'IRMSA', 'ANGLEA']

data = pd.DataFrame(data, index=np.arange(len(dataRaw)), columns=colNames)

data.TIME = pd.to_datetime(data.TIME)
offset = datetime.timedelta(hours=8)
data.TIME = data.TIME - offset


#data.TIME.hour = data.TIME.hour - 6;
days = np.zeros((len(data),3));
energyAdded = np.zeros((len(data),1));
allPF = np.zeros((len(data),1));
seshCount = np.zeros((len(data),1));

data.KVA = 3*(data.VRMSA*data.IRMSA)/1000;
data.KW = 3*(data.VRMSA*data.IRMSA)*(np.cos(data.ANGLEA*np.pi/180))/1000;
data.KVAR = 3*(data.VRMSA*data.IRMSA)*(np.sin(data.ANGLEA*np.pi/180))/1000;
result = [];

#%% Analyze Data

tST= timeit.default_timer()

for idx, row in data.iterrows():
    print(idx)
    days[idx][0] = row.TIME.dayofyear;
    days[idx][1] = row.TIME.weekday();
    days[idx][2] = row.TIME.hour;

    if row.KW != 0:
        allPF[idx] = np.cos(np.arctan(row.KVAR/row.KW))
    if idx < (len(data)-1):
        energy = data.KWH[idx+1] - data.KWH[idx]
        energyAdded[idx] = energy;

count = 1;
for idx, row in data.iterrows():
    seshCount[idx] = count;

    if idx < len(data)-1:
        if data.ANGLEA[idx] < 200 and data.ANGLEA[idx+1] > 200:
            result.append(str(idx) + ' ' + str(count) + ' ' + str(data.KWHadded[idx]) + ' ' + str(data.KWHadded[idx+1]))
            print(idx)
            count = count + 1;

#    if idx < len(data)-1:
#        if data.KWHadded[idx] < 1.0 and data.KWHadded[idx+1] > 1.0:
#            result.append(str(idx) + ' ' + str(count) + ' ' + str(data.KWHadded[idx]) + ' ' + str(data.KWHadded[idx+1]))
#            print(idx)
#            count = count + 1;


data.DAY = days[:,0];
#Return the day of the week represented by the date. Monday == 0 … Sunday == 6
data.WEEKDAY = days[:,1];
data.HOUR = days[:,2];
data.KWHadded = energyAdded;
data.PF = allPF;
data.SESSION = seshCount;

dataHead = data.head(100)

tEl = timeit.default_timer() - tST
print('Analysis Time: {0:.4f} sec'.format(tEl))

#%% Calculate Session Energy and Duration

tST= timeit.default_timer()

numSessions = int(np.max(data.SESSION));
seshEnergy = np.zeros((numSessions,3))
idx = 0;

for sesh in range(1, numSessions):

    dfTemp = data.loc[data.SESSION == sesh];

    if len(dfTemp) != 1:
        seshKWH = dfTemp.iloc[len(dfTemp)-1].KWH - dfTemp.iloc[0].KWH;

        print(sesh, ': ', seshKWH, 'kWh' )

        seshEnergy[idx][0] = seshKWH;
        seshEnergy[idx][1] = dfTemp.iloc[0].WEEKDAY;
        seshEnergy[idx][2] = len(dfTemp);
        idx += 1;


dfSeshEnergy = pd.DataFrame(seshEnergy, index=np.arange(len(seshEnergy)), columns=['KWH', 'WEEKDAY', 'TIME'])
dfSeshEnergy = dfSeshEnergy.loc[dfSeshEnergy.KWH > 0.5];

tEl = timeit.default_timer() - tST
print('Energy Session: {0:.4f} sec'.format(tEl))

#%% Plot seshEnergy Histogram

import matplotlib.pyplot as plt

#maxBin = np.ceil(np.max(seshKWH)) + 0.5;
qE_high = dfSeshEnergy['KWH'].quantile(0.9545); #remove 2 std dev outlier
qE_low = dfSeshEnergy['KWH'].quantile(1-0.9545); #remove 2 std dev outlier
seshEnergy1 = dfSeshEnergy.loc[dfSeshEnergy.TIME < qE_high];
seshEnergy1 = seshEnergy1.loc[dfSeshEnergy.TIME > qE_low];
maxVal = int(qE_high + (5 - qE_high) % 5);
minVal = int(qE_low + (5 - qE_low) % 5) - 5
binEdges = np.arange(minVal, maxVal, 1)
numBins = int(np.sqrt(len(seshEnergy1)));

n, bins, patches = plt.hist(seshEnergy1.KWH, bins=binEdges, density=True, rwidth=0.75, color='#607c8e');

plt.xlabel('Energy (kWh)')
#plt.xticks(np.arange(minVal, maxVal, 5))
plt.ylabel('Frequency')
plt.title('Energy Per Session')

print('Mean: ', np.mean(seshEnergy1.KWH), ' | Std: ', np.std(seshEnergy1.KWH))

#%% Plot seshTime Histogram

#test = dfSeshEnergy.loc[dfSeshEnergy.TIME < 30];

#val = np.max(test.TIME);
qT_high = dfSeshEnergy['TIME'].quantile(0.9545); #remove 2 std dev outlier
qT_low = dfSeshEnergy['TIME'].quantile(1-0.9545); #remove 2 std dev outlier
seshTime1 = dfSeshEnergy.loc[dfSeshEnergy.TIME < qT_high];
seshTime1 = seshTime1.loc[dfSeshEnergy.TIME > qT_low];
maxVal = int(qT_high + (5 - qT_high) % 5);
minVal = int(qT_low + (5 - qT_low) % 5) - 5
binEdges = np.arange(minVal, maxVal, 1)
numBins = int(np.sqrt(len(seshTime1)));

n, bins, patches = plt.hist(seshTime1.TIME, bins=binEdges, density=True, rwidth=0.75, color='#912727', cumulative=False);

plt.xlabel('Minutes')
#plt.xticks(np.arange(0,maxBin+1,1))
plt.ylabel('Frequency')
plt.title('Session Duration')

print('Mean: ', np.mean(seshTime1.TIME), ' | Std: ', np.std(seshTime1.TIME))

#%% Calculate minute Energy

allEnergy = list(set(data.KWH))
allEnergy.sort()
minKWH = np.zeros((len(allEnergy),1))

for i in range(len(allEnergy)-1):
    minKWH[i] = allEnergy[i+1] - allEnergy[i]


#%% Plot minKWH Histogram

import matplotlib.pyplot as plt

#maxBin = np.ceil(np.max(seshKWH)) + 0.5;
minKWH = minKWH[np.where( minKWH > 0.5 )];
maxBin = 10;
binEdges = np.arange(0,maxBin,0.5)

n, bins, patches = plt.hist(minKWH, bins=binEdges, density=True, rwidth=0.75, color='#607c8e')

plt.xlabel('Energy (kWh)')
plt.xticks(np.arange(0,maxBin+1,1))
plt.ylabel('Frequency')
plt.title('Energy Per Minute')


#%% Sessions & Energy Per Day

allDays = list(set(data.DAY))
dayKWH = np.zeros((len(allDays),1))
i = 0;

for day in allDays:
    dfTemp = data.loc[data.DAY == day];
    L = len(dfTemp);
    dayEnergy = dfTemp.KWH.iloc[L-1] - dfTemp.KWH.iloc[0];
    dayKWH[i] = dayEnergy;
    i += 1;

#%% Plot Total kWh per day Histogram

import math
import matplotlib.pyplot as plt

#plt.rcParams.update(plt.rcParamsDefault)

maxkWH = 2500; # np.max(dayKWH);

maxBin = int(math.ceil(maxkWH / 500.0)) * 500 + 500;
binEdges = np.arange(0, maxBin, 100)

n, bins, patches = plt.hist(dayKWH, bins=binEdges, density=True, rwidth=0.75, color='#607c8e', cumulative=False)

plt.xlabel('Daily Energy (kWh)')
plt.ylabel('Frequency')
plt.title('Energy Per Day')
plt.grid(False)

print('Daily Mean: ', np.mean(dayKWH) , '| Daily Std Dev: ', np.std(dayKWH))

#%% Plot Violin Plot

import seaborn as sns

#dataON = data.loc[data.KWHadded > 0.5]

ax = sns.violinplot(x='WEEKDAY', y='KWH', data=seshEnergy1)

days = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

plt.xlabel('Weekeday')
plt.xticks(np.arange(7), days)
plt.ylabel('Energy (kWh)')
plt.title('Energy Per Session')


#%% Driving Energy

# 20 driving minutes each direction from 7am to 11:59p = 17 hrs
# assume 30 min from leaving stop A to leaving stop B
stEnergy = 126; #kWh
route = 14.2; #miles
#https://www.proterra.com/performance/range/
eff = 0.5*(1.28+2.08); #kWh/mile operating efficiency
typicalDay = np.median(dayKWH)
typicalBus = typicalDay/3;

#17 hours of operation, 30 min each direction, so 17 total round trips
milesPerDay = route * 17;
kWhperRoute = route * eff;
kWhNeedPerBusDay = milesPerDay * eff;

#Assume 3 busses on route
kWhNeedPerDay = 3 * kWhNeedPerBusDay;

#%% Calculate Load Factor

#1min Peak Load
peakKW = np.max(data.KW);

#3 or 5 min average
m = 15;
peakKW = np.sum(data.KW[0:m])
for i in range(len(data)-m):
    if np.sum(data.KW[i:i+m]) > peakKW:
        peakKW = np.sum(data.KW[i:i+m])

peakKW = peakKW/m;


numDays = len(list(set(data.DAY)));
totKWH = data.KWH[len(data)-1] - data.KWH[0];
loadFactor = totKWH/(numDays*24*peakKW);

print('Load Factor: ', loadFactor);


#%% Plot hourly KW profile as average of hourly load

tST= timeit.default_timer()

#numSessions = int(np.max(data.SESSION));
hours = list(set(data.HOUR));
hrlyData = np.zeros((24,4));
hrlyLoad = np.zeros((len(hours),1));
idx = 0;

for hr in hours:

    dfTemp = data.loc[data.HOUR == hr];
    # Calculate Mean
    hrlyData[idx][0] = np.average(dfTemp.KW)
        # Calculate Median
    hrlyData[idx][0] = np.average(dfTemp.KW)


    idx += 1;

hours = np.arange(0,24)

plt.bar(hours, hrlyLoad[:,0])

plt.xlabel('Hour')
plt.xticks(np.arange(0, 24, 2))
plt.ylabel('Energy (kWh)')
plt.title('Average Energy per Hour')

tEl = timeit.default_timer() - tST
print('Energy Session: {0:.4f} sec'.format(tEl))

#%% Hourly Whisker Plots

import random
dfHrly = np.zeros((1000,24));

for hr in hours:

    dfTemp = data.loc[data.HOUR == hr];
    qT_high = dfTemp.KW.quantile(0.9545); #remove 2 std dev outlier
    qT_low = dfTemp.KW.quantile(1-0.9545); #remove 2 std dev outlier

    #dfTemp = dfTemp.KW.values
    dfTemp = dfTemp.loc[dfTemp.KW < qT_high];
    dfTemp = dfTemp.loc[dfTemp.KW > qT_low];

    tempVal = dfTemp.KW.sample(1000);

    dfHrly[:,hr] = tempVal.values;
    print(hr);

font = {'family' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

plt.figure(figsize=(16,8))
plt.boxplot(dfHrly, notch=True, showfliers=False, showmeans=True, patch_artist=True,)
#https://sites.google.com/site/davidsstatistics/home/notched-box-plots
plt.title('Daily Load Profile')
plt.xlabel('Hr')
plt.ylabel('kW')

plt.show()

qT_high = dfSeshEnergy['TIME'].quantile(0.9545); #remove 2 std dev outlier
qT_low = dfSeshEnergy['TIME'].quantile(1-0.9545); #remove 2 std dev outlier
seshTime1 = dfSeshEnergy.loc[dfSeshEnergy.TIME < qT_high];
seshTime1 = seshTime1.loc[dfSeshEnergy.TIME > qT_low];


#%% Plot Load Profile
start = '2018-01-10 00:00:00'
end = '2018-01-11 00:00:00'
dataProfile = data.loc[data.TIME > pd.to_datetime(start)];
dataProfile = dataProfile.loc[data.TIME < pd.to_datetime(end)];

plt.figure(figsize=(16,8))
plt.plot(dataProfile.TIME, dataProfile.KW)

plt.title('Daily Load Profile')
plt.xlabel('Hr')
plt.ylabel('kW')

#%%
#from matplotlib import style
#style.use('fivethirtyeight')
plt.rcParams.update(plt.rcParamsDefault)
dataTest = data;
numSessions = int(np.max(data.SESSION));

plt.ion() ## Note this correction
#fig=plt.figure()
#f, ax = plt.subplots(1)

npSesh = np.zeros((0,2))

for sesh in range(1, numSessions):

    dfTemp = dataTest.loc[dataTest.SESSION == sesh];
    dfTemp = dfTemp.loc[dfTemp.KW > 5];
    if len(dfTemp) < 40:
        x = np.arange(0,len(dfTemp));
        y = dfTemp.KW.values;

        #ax.scatter(x, y, s=4.0, alpha=0.40)
        #plt.hist2d(y)

        npTemp = np.column_stack((x,y));
        npSesh = np.vstack((npSesh, npTemp));

#%% Export data

data.to_csv('outputFile.csv')
