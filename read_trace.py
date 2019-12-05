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

dataTest = pd.read_csv('data/hdc_wkdy80.csv',index_col=[0])
dataTrn = pd.read_csv('data/hdc_wkdy20.csv',index_col=[0])

#%% Overdispersion of Data

hours = np.arange(24)
ratio = []; m = []; v = [];

for h in hours:
    dfTemp = data.loc[data.Hour==h];
    ratio.append(np.var(dfTemp.Connected)/np.mean(dfTemp.Connected))
    m.append(np.mean(dfTemp.Connected))
    v.append(np.var(dfTemp.Connected))
    
#%%plt.plot(hours, ratio)
ml = 14;    
plt.rcParams['font.family'] = "Times New Roman"
plt.figure(figsize=(4,4))
plt.scatter(m, v, s=12, label='Data')
plt.plot(np.arange(ml+1),np.arange(ml+1), lw=1, ls='dashed', c='grey', label='Poisson')
plt.legend()
plt.grid()
plt.xlim((0,ml)); plt.ylim((0,ml))
plt.xlabel('Mean'); plt.ylabel('Variance')
plt.title('Overdispersion of Data')

#%% Get 24 hour usage to show temporal stochasticity

usage24hr = {}

for h in hours:
    totSesh = np.sum(dataRaw.loc[dataRaw.Hour == h].Connected)
    usage24hr[h] = totSesh   

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
    #allAlpha = pd.concat([allAlpha, dctData[h].alpha], ignore_index=True)  
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

#%% 24 Hour Histogram Plots

fig = plt.figure(figsize=(10,6))

fig.add_subplot(211)
binsCnt = np.arange(0,np.max(allMu.values)+1)
plt.hist(allMu.values, bins=binsCnt, histtype='step', density=True, edgecolor='blue', lw=1.5, label=r'$ \mu $')
#plt.hist(allAlpha.values, bins=binsCnt, histtype='step', density=True, edgecolor='grey', lw=1.5, label=r'$ \alpha $')
plt.xlim(0,12)
plt.legend()
plt.title('All Hour Poisson Distribution Parameters')

fig.add_subplot(212)
mu = np.mean(allMu)
binsCnt = np.arange(0,16)
plt.hist(dataTest.Connected, bins=binsCnt, density=True, color='lightblue', edgecolor='white', label='Observed')
plt.hist(allYpred.values, bins=binsCnt, histtype='step', density=True, color='blue', lw=1.5, label='Predicted')
plt.hist(get_poiss_vals(mu, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='Poiss Dist')    
plt.xlim(0,12)
plt.legend()
plt.title('All Hour Poisson Prediction vs. Observed')

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
params = pd.DataFrame(index=hours, columns=['mu'])#, 'alpha'])
dim = 'y_pred'

fig, axs = plt.subplots(4, 6, figsize=(20,12), sharex=True, sharey=True) 

r,c = 0,0;

for hr in hours:     

    #kBins = 1 + 3.22*np.log(len(dctData[hr][dim])) #Sturge's Rule for Bin Count
    hists[hr] = np.histogram(dctData[hr][dim].values)#, bins=np.arange(16))        
    
    print('position', r, c)    
    #alpha = dctSmry[hr]['mean']['alpha']
    mu = dctSmry[hr]['mean']['mu']
    params.iloc[hr].mu = mu; #params.iloc[hr].alpha = alpha; 

    axs[r,c].hist(dataIN.loc[dataIN.Hour == hr].Connected, ec='white', fc='lightblue', bins=np.arange(16), density=True, label='Observed')  
    #axs[r,c].hist(get_nb_vals(mu, alpha, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='NB Dist')    
    axs[r,c].hist(get_poiss_vals(mu, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='Poiss Dist')    
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

#%% Single Hour Plot
 
#alpha = dctSmry[hr]['mean']['alpha']
mu = dctSmry[hr]['mean']['mu']
params.iloc[hr].mu = mu; #params.iloc[hr].alpha = alpha; 

plt.hist(dataIN.loc[dataIN.Hour == hr].Connected, ec='white', fc='lightblue', bins=np.arange(16), density=True, label='Observed')  
#plt.hist(get_nb_vals(mu, alpha, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='NB Dist')    
plt.hist(get_poiss_vals(mu, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='Poiss Dist')    
plt.hist(dctData[hr][dim].values, histtype='step', ec='blue', lw=1.2, bins=np.arange(16), density=True, label='Predicted') 
plt.title('Hr: ' + str(hr))

plt.ylim(0,0.25)
plt.legend()
plt.title('Poisson Trace Hr ' + str(hr))

#%% Read Results

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dirPoi = 'results/1239704_10k_Poiss_NormP'
dirNB = 'results/1239578_10k_NB_NormP'

yPred_Poi = pd.read_csv(dirPoi + '/out_yPred.csv', index_col=[0])
yPred_NB = pd.read_csv(dirNB + '/out_yPred.csv',  index_col=[0])


#%% Read / Organize yPred

yPred_Poi2 = pd.DataFrame(np.zeros((yPred_Poi.size,3)), columns=['EVs','Count','Hr'])
yPred_NB2 = pd.DataFrame(np.zeros((yPred_NB.size,3)), columns=['EVs','Count','Hr'])

for h in np.arange(24):
    r = 15*h
    yPred_Poi2.EVs[r:r+15] = np.arange(0,15)
    yPred_Poi2.Count[r:r+15] = yPred_Poi.iloc[:,[h]].values.reshape(15)
    yPred_Poi2.Hr[r:r+15] = h    

for h in np.arange(24):
    r = 15*h
    yPred_NB2.EVs[r:r+15] = np.arange(0,15)
    yPred_NB2.Count[r:r+15] = yPred_NB.iloc[:,[h]].values.reshape(15)
    yPred_NB2.Hr[r:r+15] = h  

#%% Read / Organize traces

import pandas as pd
import glob

all_files = glob.glob(dirPoi + "/*trace.csv")
li = []
qntNB_Trn = pd.DataFrame(np.zeros((24,6)), columns=['Min','25pct','Med','Mean','75pct','Max'])

for h in np.arange(24):
    filename = dirPoi + '/out_hr' + str(h) + '_trace.csv'
    df = pd.read_csv(filename, index_col=[0], header=0)     
    df['Hr'] = h
    li.append(df) 
    qntNB_Trn.iloc[h] = [np.min(df.y_pred), np.quantile(df.y_pred, 0.25), np.median(df.y_pred), np.mean(df.y_pred), np.quantile(df.y_pred, 0.75), np.max(df.y_pred)]
   
tracePoi = pd.concat(li, axis=0, ignore_index=True)
li = []
qntPoi_Trn = pd.DataFrame(np.zeros((24,6)), columns=['Min','25pct','Med','Mean','75pct','Max'])
    
for h in np.arange(24):
    filename = dirNB + '/out_hr' + str(h) + '_trace.csv'
    df = pd.read_csv(filename, index_col=[0], header=0) 
    df['Hr'] = h   
    li.append(df) 
    qntPoi_Trn.iloc[h] = [np.min(df.y_pred), np.quantile(df.y_pred, 0.25), np.median(df.y_pred), np.mean(df.y_pred), np.quantile(df.y_pred, 0.75), np.max(df.y_pred)]

traceNB = pd.concat(li, axis=0, ignore_index=True)
    
trace_Both = pd.DataFrame(np.zeros((2*len(traceNB),4)), columns=['Hr','y_pred', 'mu', 'Dist'])
trace_Both.Hr[0:len(traceNB)] = traceNB.Hr
trace_Both.Hr[len(traceNB):2*len(traceNB)] = tracePoi.Hr

trace_Both.y_pred[0:len(traceNB)] = traceNB.y_pred
trace_Both.mu[0:len(traceNB)] = traceNB.mu
trace_Both.Dist[0:len(traceNB)] = 'NegBino'

trace_Both.y_pred[len(traceNB):2*len(traceNB)] = tracePoi.y_pred
trace_Both.mu[len(traceNB):2*len(traceNB)] = tracePoi.mu
trace_Both.Dist[len(traceNB):2*len(traceNB)] = 'Poiss'

#%% Split Violin Plot

font = {'family' : 'Times New Roman',
        'size'   : 16}

plt.rc('font', **font)

plt.figure(figsize=(16,8))

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="Hr", y="Value", data=trace_yPred, 
               hue="Dist", split=True, inner="box",               
               cut = 0, scale='width', linewidth=1.25)

#plt.ylim(0, 20)
plt.title('Predictive Value Distributions')
plt.legend(title='')
plt.ylabel('EVs')

#%% Violin Plot

yLabel = 'mu'

plt.rcParams['font.family'] = "Times New Roman"
plt.figure(figsize=(20,8))

sns.violinplot(x="Hr", y=yLabel, data=tracePoi, scale='width', split=True)
plt.title('Poisson Likelihood')

#% Violin Plot Single Hour

plt.figure(figsize=(20,8))

sns.violinplot(x="Hr", y=yLabel, data=traceNB, scale='width', split=True)
plt.title('Negative Binomial Likelihood')

#%% Plot Training Quantiles

s = 10000;
trace_Smpl = trace_Both.sample(s)

font = {'family' : 'Times New Roman', 'size'   : 16}
plt.rc('font', **font)

gTrn = sns.relplot(x='Hr', y='y_pred', kind='line',
                 hue='Dist', col='Dist', #ci='sd', 
                 data=trace_Smpl)

#pltset_title('EV Arrivals')
plt.ylabel('y_pred')
plt.xticks(np.arange(0,26,2))

getTrn = pd.DataFrame(np.zeros((24,3)), columns=['Hr','Poisson''Pint','NegBino','NBint'])
getTrn.Hr = gTrn.axes.flat[0].lines[0].get_xdata()
getTrn.Poisson = gTrn.axes.flat[0].lines[0].get_ydata()    
getTrn.NegBino = gTrn.axes.flat[1].lines[0].get_ydata()

print('\nPoisson SMAPE', SMAPE(getTest[:,1],getTrn.Poisson.values))
print('NegBino SMAPE', SMAPE(getTest[:,1],getTrn.NegBino.values))

#%% Error Over Samples

err = pd.DataFrame(np.zeros((24,5)), columns=['Samples','PoiSMAPE','NbSMAPE','PoiMAPE','NbMAPE'])
i=0;
for s in [10000]:#[500,1000,5000,10000,15000,20000,25000,50000]:#,100000,150000,200000,250000,300000,350000,400000,450000]:
    trace_Smpl = trace_Both.sample(s)
    
    gTrn = sns.relplot(x='Hr', y='y_pred', kind='line',
                 hue='Dist', col='Dist', #ci='sd', 
                 data=trace_Smpl)
    
    getTrn = pd.DataFrame(np.zeros((24,5)), columns=['Hr','Poisson','Pint','NegBino','NBint'])
    getTrn.Hr = gTrn.axes.flat[0].lines[0].get_xdata()
    getTrn.NegBino = gTrn.axes.flat[0].lines[0].get_ydata()
    getTrn.Poisson = gTrn.axes.flat[1].lines[0].get_ydata()
    getTrn.NBint = np.round(gTrn.axes.flat[0].lines[0].get_ydata(),0)
    getTrn.Pint = np.round(gTrn.axes.flat[1].lines[0].get_ydata(),0)
    
    err.Samples[i] = s;
    err.PoiSMAPE[i] = SMAPE(getTest.Int, getTrn.Pint);
    err.NbSMAPE[i] = SMAPE(getTest.Int, getTrn.NBint);
    err.PoiMAPE[i] = MAPE(getTest.Int, getTrn.Pint);
    err.NbMAPE[i] = MAPE(getTest.Int, getTrn.NBint);
        
    print('\nSMAPE with ', s, 'samples.')
    print('------ SMAPE MAPE -----')
    print('Poisson ', np.round(err.PoiSMAPE[i],2), np.round(err.PoiMAPE[i],2))
    print('NegBino ', np.round(err.NbSMAPE[i],2), np.round(err.NbSMAPE[i],2))
    i+=1;    

#%% Calculate SMAPE
    
def SMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def MAPE(A, F):
    return 100/len(A) * np.sum(np.abs(F - A) / np.abs(A))
    

#%% Aggregate of ALL Test Data

#dataTest = pd.read_csv('data/hdc_wkdy80.csv', index_col=[0])

import random    
daysInTest = list(set(dataTest.DayYr))
daysInTrn = list(set(dataTrn.DayYr))

daysIn = random.choices(daysInTest, k=2)


font = {'family' : 'Times New Roman', 'size'   : 16}
plt.rc('font', **font)

# Aggrgate of Test Data
#gTest = sns.relplot(x='Hour', y='Connected', kind='line', color='0.3', markers=True,
#                 data=dataTest)

# n Days of Test Data
gTest = sns.relplot(x='Hour', y='Connected', kind='line', color='0.3', markers=True,
                 data=dataTest.where(dataTest.DayYr.isin(daysIn)) )

plt.title('Test Value Spread')
#plt.legend(title='')
plt.ylabel('y_test')
plt.xticks(np.arange(0,26,2))

getTest = pd.DataFrame(np.zeros((24,3)), columns=['Hr','Value','Int'])
i=0;
for ax in gTest.axes.flat:
    print (ax.lines)
    for line in ax.lines:
        getTest.Hr = line.get_xdata();
        getTest.Value = line.get_ydata();
    i+=1;
    
getTest.Int = np.round(getTest.Value,0)

#%%

plt.scatter(getTest[:,0],getTest[:,1], marker='x', color='k', label='Test')

gTrn = sns.relplot(x='Hr', y='y_pred', kind='line',
                 hue='Dist', col='Dist', #ci='sd', 
                 data=trace_Smpl)

#%% Fill Plot 

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(24)
q = 0.10;

qnt_Trn = pd.DataFrame(np.zeros((24,8)), columns=['Min','25pct', 'q1', 'Med','Mean', 'q2','75pct','Max'])
dirPoi = 'results/1239704_10k_Poiss_NormP'; dirNB = 'results/1239578_10k_NB_NormP';

for h in np.arange(24):
    filename = dirNB + '/out_hr' + str(h) + '_trace.csv'
    df = pd.read_csv(filename, index_col=[0], header=0)     
    qnt_Trn.iloc[h] = [np.min(df.y_pred), np.quantile(df.y_pred, 0.25), np.quantile(df.y_pred, 0.50-q), np.median(df.y_pred), 
                  np.mean(df.y_pred), np.quantile(df.y_pred, 0.50+q), np.quantile(df.y_pred, 0.75), np.max(df.y_pred)]
   
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, qntNB_Trn.Med, x, qnt_Trn['q2'], color='black', lw=0.5)
ax.fill_between(x, qnt_Trn.Med, qnt_Trn['q2'], where=qnt_Trn['q2']>qnt_Trn.Med, 
                facecolor='orange', alpha=0.1)
ax.plot(x, qntNB_Trn.Med, x, qnt_Trn['q1'], color='black', lw=0.5)
ax.fill_between(x, qnt_Trn.Med, qnt_Trn['q1'], where=qnt_Trn['q1']<qnt_Trn.Med, 
                facecolor='orange', alpha=0.1)

ax.plot(x, qntNB_Trn.Med, '--', c='orange', lw=2)

plt.scatter(x,getTest[:,2], marker='x', color='k', label='Test')

#ax.set_title('NegBino')
ax.set_title('Poisson')
ax.set_xticks(np.arange(0,26,2))

#%%

import pymc3 as pm

# Convert categorical variables to integer
hr_idx = data.Hour.values
hrs = np.arange(24)
n_hrs = len(hrs)

#% Hierarchical Modeling
with pm.Model() as model:
    hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=20)
    hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=20)
    #hyper_alpha_mu = pm.Normal('hyper_alpha_mu', mu=hr_std)

    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=20)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=20)
    #hyper_mu_mu = pm.Normal('hyper_mu_mu', mu=hr_mean)

    alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd, shape=n_hrs)
    mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd, shape=n_hrs)
    #alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd)
    #mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd)

    y_obs = data.Connected.values

    #y_est = pm.Poisson('y_est', mu=mu[hr_idx], observed=y_obs)
    #y_pred = pm.Poisson('y_pred', mu=mu[hr_idx], shape=data.Hour.shape)

    y_est = pm.NegativeBinomial('y_est', mu=mu[hr_idx], alpha=alpha[hr_idx], observed=y_obs)
    y_pred = pm.NegativeBinomial('y_pred', mu=mu[hr_idx], alpha=alpha[hr_idx], shape=data.Hour.shape)

#%%    

trace1 = pm.load_trace('/results/667548_20k_poolNB/out_smpls20000.trace')


#%%

from scipy.stats import nbinom
import matplotlib.pyplot as plt

result_NB_20k = pd.read_excel('results/NB_20kpool.xlsx');

result_NB_20k['r'], result_NB_20k['p'] = convert_params(result_NB_20k['alpha'],result_NB_20k['mu']) 

#%%
x = np.arange(0, 10, 1)

for i in [12]:
    
    r = result_NB_20k['r'].at[i]
    p = result_NB_20k['p'].at[i]

    plt.bar(x, nbinom.pmf(x, r, p), label='nbinom pmf')
#ax.vlines(x, 0, nbinom.pmf(x, r, p), colors='b', lw=5, alpha=0.5)

#%%
rdmNB = nbinom.rvs(r, p, size=100000)
plt.hist(rdmNB, density=True)

#%%

def convert_params(mu, alpha):
    """ 
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    Parameters
    ----------
    mu : float 
       Mean of NB distribution.
    alpha : float
       Overdispersion parameter used for variance calculation.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    var = mu + alpha * mu ** 2
    p = (var - mu) / var
    r = mu ** 2 / (var - mu)
    return r, p


#%%
import XlsxWriter

writer = pd.ExcelWriter('out_trace.xlsx', engine='xlsxwriter')

trace_data.to_excel(writer, sheet_name='trace_data01')

writer.save()