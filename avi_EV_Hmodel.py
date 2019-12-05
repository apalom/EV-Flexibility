# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:36:59 2019

@author: Alex Palomino
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm

# Get SLC 2018 Weekday Data
dataRaw = pd.read_excel('data/dfSLC_Wkdy_2018.xlsx')

#%% Get Station Names and  
stations = pd.read_csv('data/Station-table-all-columns-20191104.csv')
stations = stations.loc[stations.City == 'Salt Lake City']
stations = stations.reset_index(drop=True)

dataStations = pd.DataFrame(np.zeros((len(stations),2)), columns=['Station','Site'])
dataStations.Station = stations['Station Name']
dataStations.Site = stations['Location']

#%% Build Work and Public Datasets
stationsWork = list(dataStations.Station.loc[dataStations.Site == 'Workplace'])
stationsPublic = list(dataStations.Station.loc[dataStations.Site == 'Public'])

dfWork = dataRaw.loc[dataRaw['Station Name'].isin(stationsWork)]
dfPublic = dataRaw.loc[dataRaw['Station Name'].isin(stationsPublic)]

#%% Create Hour_DayCnt_DayYr_Connected data

df = dfWork
daysIn = list(set(df.dayCount))
daysIn.sort()

dfDays = pd.DataFrame(np.zeros((24,len(set(df.dayCount)))), 
                    index=np.arange(0,24,1), columns=daysIn)

for d in df.dayCount:
    print('Day: ', d)
    dfDay = df[df.dayCount == d]
    cnct = dfDay.StartHr.value_counts()
    cnct = cnct.sort_index()
    
    dfDays.loc[:,d] = dfDay.StartHr.value_counts()
    dfDays.loc[:,d] = np.nan_to_num(dfDays.loc[:,d])

daysIn = dfDays.shape[1]
dfHrCnctd = pd.DataFrame(np.zeros((24*daysIn,4)), columns=['Hour','DayCnt','DayYr','Connected'])

r = 0;
d = 0;

for j in list(dfDays):   
    
    dfHrCnctd.Hour.iloc[r:r+24] = np.linspace(0,23,24);
    dfHrCnctd.DayCnt.iloc[r:r+24] = np.repeat(d, 24);
    dfHrCnctd.DayYr.iloc[r:r+24] = j;
    dfHrCnctd.Connected[r:r+24] = dfDays[j]
    
    d += 1;
    r += 24;

dfHrCnctd_work = dfHrCnctd    
#dfHrCnctd.to_excel("data/dfPublic2018wkdy.xlsx")    

#%% Load Data and Define Hierarchical Model 
  
obsVals = dfHrCnctd_work['Connected'].values

# Define indices for hourly partial pooling
le = preprocessing.LabelEncoder()
hrs_idx = le.fit_transform(dfHrCnctd_work['Hour'])
hrs = le.classes_
n_hrs = len(hrs)
    
with pm.Model() as EVmodel:
    
    # Hyper-Priors
    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=10)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=10) 
    
    # Priors    
    mu = pm.Gamma('mu', mu=hyper_mu_mu, 
                        sigma=hyper_mu_sd,
                        shape=n_hrs)    
    
    # Data Likelihood
    y_like = pm.Poisson('y_like', 
                       mu=mu[hrs_idx], 
                       observed=obsVals)    
    
pm.model_to_graphviz(EVmodel)

#% Hierarchical Model Inference

# Setup vars
smpls = 2500; tunes = 500; ch = 4;
    
# Print Header
print('Work Charging')
print('Params: samples = ', smpls, ' | tune = ', tunes)
        
with EVmodel:
    trace = pm.sample(smpls, tune=tunes, chains=ch, cores=1)
    
    ppc_work = pm.sample_posterior_predictive(trace)
    #pm.traceplot(trace)                  

out_smryWork = pd.DataFrame(pm.summary(trace))  

#%% Posterior Predictive Results

obsVals = dfHrCnctd_public['Connected'].values
hrs_idx = le.fit_transform(dfHrCnctd_public['Hour'])

ppc_Sample = pd.DataFrame(ppc_public['y_like'])
ppcVals = np.reshape(ppc_Sample.sample(smpls*ch).values, (smpls*ch*len(obsVals),1))
ppc_Sample = pd.DataFrame(ppc_public['y_like'].transpose())
ppc_Sample['Hr'] = hrs_idx 
ppcHrs = np.tile(hrs_idx,ppc_Sample.shape[1]-1)
ppc_Vals = pd.DataFrame(ppcVals, columns=['Connected'])
ppc_Vals['Hr'] = ppcHrs
ppc_publicVal.sample(smpls*ch).reset_index(drop=True).to_excel("results/forAvi/ppc_publicVal.xlsx") 

ppc_publicVal = ppc_Vals

#%% ppc Hourly Plot Histograms

import seaborn as sns
sns.set(style="whitegrid", font='Times New Roman', 
        font_scale=1.75)
              
#fig, axs = plt.subplots(4, 6, figsize=(20,12), sharex=True, sharey=True) 
r,c = 0,0;

probWork = pd.DataFrame(np.zeros((24,16)))
probPublic = pd.DataFrame(np.zeros((24,16)))

for h in hrs:     
    print('Hr: ', h)
    #smplWork = ppc_workVal.loc[ppc_workVal.Hr==h].sample(5000)
    probWork.loc[h] = np.histogram(ppc_workVal.loc[ppc_workVal.Hr==h].sample(5000).Connected.values,
                                    bins=np.arange(17), density=True)[0]
    #axs[r,c].hist(smplWork.Connected, ec='white', fc='lightblue', 
    #               bins=np.arange(16), density=True, label='Workplace'                   )  
    #smplPublic = ppc_publicVal.loc[ppc_publicVal.Hr==h].sample(5000)
    probPublic.loc[h] = np.histogram(ppc_publicVal.loc[ppc_publicVal.Hr==h].sample(5000).Connected.values,
                                     bins=np.arange(17), density=True)[0]
    #axs[r,c].hist(smplPublic.Connected, ec='white', fc='orange', alpha=0.3,
    #               bins=np.arange(16), density=True, label='Public')      
    #axs[r,c].set_title('Hr: ' + str(int(h)))
    
    # Subplot Spacing
#    c += 1
#    if c >= 6:
#        r += 1;
#        c = 0;
#        if r >= 4:
#            r=0;
#    print(r,c)
#  
#fig.tight_layout()
#fig.suptitle('Hourly Histogram Arrival Prediction', y = 1.02)
##xM, bS = int(np.max(dctData[hr][dim])), 4
#xM, bS = 20, 4
#plt.xlim(0,xM)
#plt.xticks(np.arange(0,xM+bS,bS))
#plt.ylim(0,0.4)
#plt.legend()
#plt.show()   