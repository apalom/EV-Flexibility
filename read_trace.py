# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:53:36 2019

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import nbinom, gamma, poisson

#%% Read-In Observed and Test80 Data

dataIN = pd.read_csv('data/hdc_wkdy80.csv', index_col=[0])
#dataTest = pd.read_csv('data/hdc_wkdy_TEST80.csv', index_col=[0])

daysTest = np.random.choice(int(np.max(dataTest.DayCnt)), 10)
dfTempTest = dataTest.loc[dataTest.DayCnt.isin(daysTest)]

#%% Read Trace Plot Mu

dctData = {}
dctSmry = {}
hours = np.arange(0,24)

allMu = pd.DataFrame(np.zeros((0,0)))
allAlpha = pd.DataFrame(np.zeros((0,0)))
allYpred = pd.DataFrame(np.zeros((0,0)))
# NegBino Data Results
path = 'results/1199600_10k_Poiss_hdc-wkdy20/out_hr'

for h in hours:
    
    # Read in ALL TRACE data
    name = path+str(h)+'_trace.csv'
    dctData[h] = pd.read_csv(name, index_col=[0])
    dctData[h]['hr'] = h
    allMu = pd.concat([allMu, dctData[h].mu], ignore_index=True)    
    allAlpha = pd.concat([allAlpha, dctData[h].alpha], ignore_index=True)  
    allYpred = pd.concat([allYpred, dctData[h].y_pred], ignore_index=True)    
    
    # Read in summary trace data
    name = path+str(h)+'_smry.csv'
    dctSmry[h] = pd.read_csv(name, index_col=[0])

dctData_All = pd.DataFrame(columns=list(dctData[0]))
dctSmry_yPred = pd.DataFrame(columns=list(dctSmry[0]))
for h in hours:
    dctData_All = dctData_All.append(dctData[h])
    dctSmry_yPred.loc[h] = dctSmry[h].loc['y_pred']

#%% Histogram Plots

fig = plt.figure(figsize=(10,6))

fig.add_subplot(211)
binsCnt = np.arange(0,np.max(allMu.values)+1)
plt.hist(allMu.values, bins=binsCnt, histtype='step', density=True, edgecolor='blue', lw=1.5, label=r'$ \mu $')
plt.hist(allAlpha.values, bins=binsCnt, histtype='step', density=True, edgecolor='grey', lw=1.5, label=r'$ \alpha $')
#plt.xlim(0,25)
plt.legend()
plt.title('All Hour NegBino Distribution Parameters')

fig.add_subplot(212)
binsCnt = np.arange(0,16)
plt.hist(dataTest.Connected, bins=binsCnt, density=True, color='lightblue', edgecolor='white', label='Observed')
plt.hist(allYpred.values, bins=binsCnt, histtype='step', density=True, color='blue', lw=1.5, label='Predicted')
plt.xlim(0,16)
plt.legend()
plt.title('All Hour NegBino Prediction vs. Observed')

plt.tight_layout()

#%%

fig = plt.figure(figsize=(6,4))
plt.grid(color='lightgrey', linewidth=0.30)

for d in daysTest:
    plt.plot(hours,dfTempTest.loc[dfTempTest.DayCnt == d].Connected, '.')

plt.plot(dctSmry_yPred['mean'])

plt.xticks(hours)    
plt.tight_layout()

#%% Hourly Plot Histograms
              
hours = np.arange(0,24)
hists = {}
params = pd.DataFrame(index=hours, columns=['mu', 'alpha'])
dim = 'y_pred'

fig, axs = plt.subplots(4, 6, figsize=(12,8), sharex=True, sharey=True) 

r,c = 0,0;

for hr in hours:     

    #kBins = 1 + 3.22*np.log(len(dctData[hr][dim])) #Sturge's Rule for Bin Count
    hists[hr] = np.histogram(dctData[hr][dim].values, bins=np.arange(16))        
    
    print('position', r, c)    
    alpha = dctSmry[hr]['mean']['alpha']
    mu = dctSmry[hr]['mean']['mu']
    params.iloc[hr].mu = mu; params.iloc[hr].alpha = alpha; 

    axs[r,c].hist(dataIN.loc[dataIN.Hour == hr].Connected, ec='white', fc='lightblue', bins=np.arange(16), density=True, label='Observed')  
    axs[r,c].hist(get_nb_vals(mu, alpha, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='NB Dist')    
    #axs[r,c].hist(get_poiss_vals(mu, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='Poiss Dist')    
    axs[r,c].hist(dctData[hr][dim].values, histtype='step', ec='blue', lw=1.2, bins=np.arange(16), density=True, label='Predicted') 
    axs[r,c].set_title('Hr: ' + str(hr))
    
    # Subplot Spacing
    c += 1
    if c >= 6:
        r += 1;
        c = 0;
        if r >= 4:
            r=0;
  
fig.tight_layout()
fig.suptitle('Hourly Histogram: '+ dim, y = 1.02)
#xM, bS = int(np.max(dctData[hr][dim])), 4
xM, bS = 20, 4
plt.xlim(0,xM)
plt.xticks(np.arange(0,xM+bS,bS))
plt.ylim(0,0.4)
plt.legend()
plt.show()

# Calculate Errors

errHr = pd.DataFrame(index=hours,columns=['Err']);
for hr in hours:
    obs = np.histogram(dataIN.loc[dataIN.Hour == hr].Connected, bins=np.arange(16), density=True)[0]
    pred = np.histogram(dctData[hr][dim].values, bins=np.arange(16), density=True)[0] 
    errHr.iloc[hr].Err = np.abs(obs-pred)
    errHr.iloc[hr].Err = errHr.iloc[hr].Err[~np.isnan(errHr.iloc[hr].Err)]
    errHr.iloc[hr].Err = errHr.iloc[hr].Err[~np.isinf(errHr.iloc[hr].Err)]
    errHr.iloc[hr].Err = np.mean(errHr.iloc[hr].Err)

#%% Hourly Scatterplot Jitter
     
lm = sns.stripplot(x='hr', y='y_pred', data=dctData_All.sample(4800),   
          size=4, alpha=.25, jitter=True, edgecolor='none')

axes = lm.axes
axes.set_ylim(0,25)
axes.set_title('Scatter Plot System')

#%% Line Plt from Summary

fig = plt.figure(figsize=(6,4))
plt.grid(color='lightgrey', linewidth=0.30)
plt.plot(dctSmry_yPred['mean'])
plt.plot(dctSmry_yPred['hpd_2.5'])
plt.plot(dctSmry_yPred['hpd_97.5'])

#%% Plot Gelman-Rubin Stat

# Read in hourly data seperated by CSV sheet
hours = np.arange(0,24)

fig = plt.figure(figsize=(6,4))
plt.grid(color='lightgrey', linewidth=0.30)

for h in hours:        
    plt.scatter(h, dctSmry[h]['Rhat']['y_pred'])

plt.xticks(np.arange(0,26,2))
#plt.title('Gelman Rubin Statistic: Rhat')
plt.title(r'Gelman-Rubin Statistic $\hat{R}$')
plt.xlabel('Hour')


#%% Pair Grid Data

for h in [8]:   
    
    g = sns.PairGrid(dctData[h], vars=["y_pred", "mu", "alpha"])
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, n_levels=3)


#%% Effective Sample Size Function
# http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-9.html

dataX = pd.read_csv('data/hr_day_cncted_wkdy.csv')
dataX = dataX.Connected

def effectiveSampleSize(data, stepSize = 1) :
  """ Effective sample size, as computed by BEAST Tracer."""
  samples = len(data)

  assert len(data) > 1,"no stats for short sequences"
  
  maxLag = min(samples//3, 1000)

  gammaStat = [0,]*maxLag
  #varGammaStat = [0,]*maxLag

  varStat = 0.0;

  if type(data) != np.ndarray :
    data = np.array(data)

  normalizedData = data - data.mean()
  
  for lag in range(maxLag) :
    v1 = normalizedData[:samples-lag]
    v2 = normalizedData[lag:]
    v = v1 * v2
    gammaStat[lag] = sum(v) / len(v)
    #varGammaStat[lag] = sum(v*v) / len(v)
    #varGammaStat[lag] -= gammaStat[0] ** 2

    # print lag, gammaStat[lag], varGammaStat[lag]    
    if lag == 0 :
      varStat = gammaStat[0]
    elif lag % 2 == 0 :
      s = gammaStat[lag-1] + gammaStat[lag]
      if s > 0 :
         varStat += 2.0*s
      else :
        break
      
  # standard error of mean
  # stdErrorOfMean = Math.sqrt(varStat/samples);

  # auto correlation time
  act = stepSize * varStat / gammaStat[0]

  # effective sample size
  ess = (stepSize * samples) / act

  return ess

esx = effectiveSampleSize(dataX)

print("Effective Size for x: ", esx, " of ", len(dataX), " samples, rate of", esx/len(dataX)*100, "%.")

    
#%% Plot Hourly Value Distributions
# https://docs.pymc.io/notebooks/GLM-negative-binomial-regression.html

def get_nb_vals(mu, alpha, size):
    """Generate negative binomially distributed samples by drawing a sample from a gamma 
    distribution with mean `mu` and shape parameter `alpha', then drawing from a Poisson
    distribution whose rate parameter is given by the sampled gamma variable."""    
    g = stats.gamma.rvs(alpha, scale=mu / alpha, size=size)
    return stats.poisson.rvs(g)

def get_poiss_vals(mu, size):
    """Generate poisson distributed samples."""    
    return stats.gamma.rvs(mu, size=size)


#%%
#mu = trace_smry['mean']['mu']
#alpha = trace_smry['mean']['alpha']
hr = 8
alpha = dctSmry[hr-1]['mean']['alpha']
mu = dctSmry[hr-1]['mean']['mu']

plt.hist(dataTest.Connected.loc[dataTest.Hour == hr], ec='white', fc='lightblue', bins=np.arange(16), density=True, label='Observed') 
plt.hist(get_nb_vals(mu, alpha, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='NB Dist')
plt.hist(dctData[hr][dim].values, histtype='step', ec='blue', lw=1.2, bins=np.arange(16), density=True, label='Predicted') 
#plt.set(xticks=np.arange(0,26,2), xlim=[-1, 25])
plt.ylim(0,0.4)
plt.legend()
plt.title('NegBino Trace Hr ' + str(hr))

#%%
import XlsxWriter

writer = pd.ExcelWriter('out_trace.xlsx', engine='xlsxwriter')

trace_data.to_excel(writer, sheet_name='trace_data01')

writer.save()