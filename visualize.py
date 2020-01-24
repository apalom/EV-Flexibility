# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:01:18 2020

@author: Alex
"""

import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

days = int(max(dfSLC_aggData.DayCnt)+1);

per = '15min';

if per == '15min':
    ppD = 96;
elif per == '1hr':
    ppD = 24;
    
cnctDays = pd.DataFrame(np.zeros((ppD,days)))

for j in range(days):
    cnctDays[j] = dfSLC_aggData.Connected[j*ppD:j*ppD+ppD].values


#%% Plot Each Day

for j in range(days):
    plt.plot(np.arange(0,periods), cnctDays[j], alpha=0.10)

plt.ylabel('Count')
plt.xlabel('Time')
plt.xticks(np.arange(0,104,8))
plt.title('EVs Connected Per Day')

#%% Plot Aggregate Hist
    
sns.set_style("whitegrid")
ax = sns.distplot(dfSLC_aggData.Connected, bins=np.arange(0,max(dfSLC_aggData.Connected)))

ax.set_title("EVs Connected")

ax.set(xlabel='Count', ylabel='Density')
plt.show()

#%% Fit Poisson

import scipy.stats as st

means = np.zeros((periods));
stdDevs = np.zeros((periods));

for i in range(periods):
    means[i] = np.mean(cnctDays.iloc[i]);
    stdDevs[i] = np.std(cnctDays.iloc[i]);
    
plt.scatter(means, stdDevs)
plt.xlim(0,18)
plt.ylim(0,18)

#%% Binned Least Squares Curve Fit

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial

# get poisson deviated random numbers
h = 8*4;
data = cnctDays.iloc[h];
most = np.max(data);

# the bins should be of integer width, because poisson is an integer distribution
entries, bin_edges, patches = plt.hist(data, bins=int(most), range=[-0.5, most+0.5], density=True, alpha=0.5)

# calculate binmiddles
bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

# poisson function, parameter lamb is the fit parameter
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

# fit with curve_fit
parameters, cov_matrix = curve_fit(poisson, bin_middles, entries) 

LSE = 100*np.round(sum((poisson(np.arange(int(most)), *parameters) - entries)**2),5);

# plot poisson-deviation with fitted parameter
x_plot = np.linspace(0, most, 1000)

plt.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2, color='b')
plt.text(most-most*0.20, 0.11, 'LSE : ' +str(LSE)+ '%')

plt.show()

#%% Calculate All Poisson Fit Parameters and LSE

# poisson function, parameter lamb is the fit parameter
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

paramFit = pd.DataFrame(np.zeros((periods,3)), columns=['Param','StdDev','LSE'])

# calculate across all time periods
for h in range(periods):
    data = cnctDays.iloc[h];
    most = np.max(data);

    entries, bin_edges = np.histogram(data, bins=int(most), range=[-0.5, most+0.5], density=True);
    # calculate binmiddles
    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

    print('---', h)
    
    # fit with curve_fit
    parameter, cov_matrix = curve_fit(poisson, bin_middles, entries);
    
    LSE = 100*np.round(sum((poisson(np.arange(int(most)), *parameter) - entries)**2),5);
    
    paramFit.iloc[h] = [parameter, np.std(data), LSE]
    
#%% Plot Single EV Connected

check = np.max(dfSLC_aggData.DayCnt);
numDays = 5; rday = check;
while rday+numDays > check:    
    rday = random.randint(0, np.max(dfSLC_aggData.DayCnt)); 
plt.figure(figsize=(15, 3))
plt.plot(dfSLC_aggData.Connected[rday*288:rday*288+numDays*288])
    
    