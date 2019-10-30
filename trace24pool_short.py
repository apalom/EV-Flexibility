# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:53:26 2019

@author: Alex Palomino
"""

#~/VENV3.6.3/bin/

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm

#%%  Import Data

data = pd.read_csv('hdc_wkdy20.csv',  index_col='Idx');

# Convert categorical variables to integer
hrs_idx = data['Hour']
hrs = np.arange(24)
n_hrs = len(hrs)

dataHrly = pd.DataFrame(np.zeros((len(hrs),2)), columns=['mu', 'sd'])
for h in hrs:
    temp = data.loc[data.Hour == h]
    dataHrly.mu.at[h] = np.mean(temp.Connected)
    dataHrly.sd.at[h] = np.std(temp.Connected)

agg = [np.mean(data.Connected), np.std(data.Connected)]

#%% Load Data and Setup Hierarchical Model 
  
data = pd.read_csv('hdc_wkdy20.csv',  index_col='Idx');

connected = data['Connected'].values
# Convert categorical variables to integer
hrs_idx = data['Hour']
hrs = np.arange(24)
n_hrs = len(hrs)
    
with pm.Model() as EVpooling:
    
    # Hyper-Priors
    hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=40)
    hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=40)

    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=40)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=40) 
    
    # Priors
    alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, 
                              sigma=hyper_alpha_sd, 
                              shape=n_hrs)
    
    mu = pm.Gamma('mu', mu=hyper_mu_mu, 
                        sigma=hyper_mu_sd,
                        shape=n_hrs)    

    #Data Likelihood
#    y_like = pm.NegativeBinomial('y_like', 
#                                 mu=mu[hrs_idx], 
#                                 alpha=alpha[hrs_idx], 
#                                 observed=connected)
    y_like = pm.Poisson('y_like', 
                            mu=mu[hrs_idx],                                   
                            observed=connected)
     
#%% Hierarchical Model Inference

# Setup vars
smpls = 5000; tunes = 5000; target = 0.90;    
    
# Print Header
print('\n Running ', str(datetime.now()))
#print('hdc_wkdy20.csv | NB with Uniform Hyper-Prior')
print('hdc_wkdy20.csv | Poisson with Uniform Hyper-Prior')
print('Params: samples = ', smpls, ' | tune = ', tunes, ' | target = ', target, '\n')
        
with EVpooling:
    trace = pm.sample(smpls, tune=tunes, cores=1, target_accept=target)   

#%% Load Trace
with EVpooling:
    traceNB = pm.load_trace('results/pool/pool_NB_5k_pt9.trace')

print('NB Likelihood - 5000 sample - 5000 tune')

out_smryPoi = pd.DataFrame(pm.summary(traceNB))  
out_tracePoi = pd.DataFrame.from_dict(list(traceNB)) 
    
#%% Save Outputs
#model_to_graphviz(EVpooling)    

out_smry = pd.DataFrame(pm.summary(tracePoi))   

out_smry.to_excel('/results/pool/pool_NB_5k_pt9.xlsx')
pm.save_trace(tracePoi, '/results/pool/pool_NB_5k_pt9.trace')
    
#%% Forestplot

plt.figure(figsize=(6,16))
pm.forestplot(tracePoi, var_names=['mu'])

#%% Calculate SMAPE
    
def SMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def MAPE(A, F):
    return 100/len(A) * np.sum(np.abs(F - A) / np.abs(A))

#%% Prep sns.relplot
    
trace_smry = pd.DataFrame(np.zeros((2*24,5)), columns=['Hr', 'mu', 'alpha', 'sd', 'Dist'])
trace_smry.Hr[0:24] = np.arange(24)
trace_smry.Hr[24:2*24] = np.arange(24)

trace_smry.mu[0:24] = out_smryNB.tail(24)['mean']
trace_smry.alpha[0:24] = out_smryNB['mean'][4:28]
trace_smry.sd[0:24] = out_smryNB.tail(24).sd
trace_smry.Dist[0:24] = 'NegBino'

trace_smry.mu[24:2*24] = out_smryPoi.tail(24)['mean']
trace_smry.sd[24:2*24] = out_smryPoi.tail(24).sd
trace_smry.Dist[24:2*24] = 'Poiss'

#%% Prep Trace Both

trace_both = pd.DataFrame(np.zeros((2*24*5000, 3)), columns=['Hour','mu','Dist'])
trace_both.Hour = np.tile(np.arange(24),10000)
trace_both.Dist[0:120000] = 'NegBino'
trace_both.Dist[120000:2*120000] = 'Poisson'

for idx, row in out_traceNB.iterrows():    
    trace_both.mu[idx*24:idx*24+24] = out_traceNB.mu.at[idx]

for idx, row in out_tracePoi.iterrows():    
    trace_both.mu[120000+idx*24:120000+idx*24+24] = out_tracePoi.mu.at[idx]

#%% Split Violin Plot

import seaborn as sns
sns.set(style="whitegrid", font='Times New Roman', 
        font_scale=1.75)

plt.figure(figsize=(16,8))

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="Hour", y="mu", data=trace_both, 
               hue="Dist", split=True, inner="box")#,               
               #cut=0)#, scale='width', linewidth=1.25)

#plt.ylim(0, 20)
plt.title('Predictive Value Distributions')
plt.legend(title='')
plt.ylabel('EV Arrivals')

#%%

def convert_NB(mu, alpha):
    var = mu + alpha * mu ** 2
    p = (var - mu) / var
    r = mu ** 2 / (var - mu)
    n = mu*p/(1-p)
    return r, p, n

s = 1000;
NB_rp = np.zeros((24,3))
NB_rvs = np.zeros((24,s))
Poi_rvs = np.zeros((24,s))

for h in hrs:
    mu1 = out_smryNB.tail(24)['mean'][h]
    alpha1 = out_smryNB['mean'][4:28][h]
    r, p, n = convert_NB(mu1, alpha1)
    print(p,n)
    NB_rp[h:] = [r,p,n] 
    NB_rvs[h:] = stats.nbinom.rvs(n=n, p=p, size=s)
    Poi_rvs[h:] = stats.poisson.rvs(out_smryPoi.tail(24)['mean'][h])

dist_models = pd.DataFrame(np.zeros((2*24*s,3)), columns=['Hr','Val','Dist'])

dist_models.Hr = np.tile(np.arange(24),2*s)
dist_models.Dist[0:s*24] = 'NegBino'
dist_models.Dist[s*24:2*s*24] = 'Poisson'

for i in range(s):   
    a = i*24; b = i*24+24; mid = s*24;
    c = mid+i*24; d = mid+i*24+24;
    dist_models.Val[a:b] = NB_rvs[:,i]
    dist_models.Val[i*24+24:i*24+24+24] = Poi_rvs[:,i]

#%% SNS Relationship Plot

plt.figure(figsize=(16,8))
g_test = sns.relplot(x='Hour', y='mu', kind='line',
                 hue='Dist', col='Dist', ci='sd', 
                 data=trace_both)

plt.scatter(10,2)

plt.xticks(np.arange(0,24,2))


#%% Get Test Data

import seaborn as sns

dataTrn = data
dataTest = pd.read_csv('data/hdc_wkdy80.csv', index_col=[0])

import random    
daysInTest = list(set(dataTest.DayYr))
daysInTrn = list(set(dataTrn.DayYr))

testDays = 5;
daysIn = random.choices(daysInTest, k=5)

font = {'family' : 'Times New Roman', 'size'   : 16}
plt.rc('font', **font)

# Aggrgate of Test Data
#gTest = sns.relplot(x='Hour', y='Connected', kind='line', color='0.3', markers=True,
#                 data=dataTest)

# n Days of Test Data
gTest = sns.relplot(x='Hour', y='Connected', kind='line', color='0.3', markers=True,
                 data=dataTest.where(dataTest.DayYr.isin(daysIn)) )

plt.title('Test Values')
#plt.legend(title='')
plt.ylabel('mu')
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

#%% Error Over Samples

err = pd.DataFrame(np.zeros((24,5)), columns=['Samples','PoiSMAPE','NbSMAPE','PoiMAPE','NbMAPE'])
i=0;
for s in [500,5000,25000,50000]:#,100000,150000,200000,250000,300000,350000,400000,450000]:
    trace_Smpl = trace_both.sample(s)
    
    gTrn = sns.relplot(x='Hour', y='mu', kind='line',
                 hue='Dist', col='Dist', ci='sd', 
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

#%% Get Training Data from Trace_Summary

getTrn = pd.DataFrame(np.zeros((24,5)), columns=['Hr','Poisson','Pint','NegBino','NBint'])
getTrn.Hr = np.arange(24)
getTrn.NegBino = trace_smry.mu.where(trace_smry=='NegBino')
getTrn.NBint = np.round(getTrn.NegBino,0)
getTrn.Poisson = trace_smry.mu.where(trace_smry=='Poisson')
getTrn.Pint = np.round(getTrn.Poisson,0)

#%% Calculate Error from Trace_Summary

err = pd.DataFrame(np.zeros((1,5)), columns=['Samples','PoiSMAPE','NbSMAPE','PoiMAPE','NbMAPE'])
err.Samples[0] = s;
err.PoiSMAPE[0] = SMAPE(getTest.Int, getTrn.Pint);
err.NbSMAPE[0] = SMAPE(getTest.Int, getTrn.NBint);
err.PoiMAPE[0] = MAPE(getTest.Int, getTrn.Pint);
err.NbMAPE[0] = MAPE(getTest.Int, getTrn.NBint);

#%% Calculate Number of Errors

errNet = []; errNetAvg = []; errSMAPE = [];
errNetHr = np.zeros((24,209));
daysTrn = np.int(np.max(dataTest.DayCnt))

for idx in range(daysTrn):    
    val = (np.abs(dataTest.loc[dataTest.DayCnt == idx].Connected.values - getTrn.NBint.values));
    errNet = np.append(errNet,val);
    errNetAvg = np.append(errNetAvg, np.average(val));
    errNetHr[:,idx] = val;
    errSMAPE.append(SMAPE(dataTest.loc[dataTest.DayCnt == idx].Connected.values, getTrn.NBint.values))
    print(idx, np.round(np.average(val),2))    

errNetHrAvg = np.mean(errNetHr)
errNetHr = pd.DataFrame(errNetHr)

#%%
errAbs_qnt = pd.DataFrame(np.zeros((24,3)), columns=['25pct', 'Med', '75pct'])
errScale = pd.DataFrame(np.zeros((24,3)), columns=['MAE', 'MSE', 'RMSE'])

for h in hrs:
    errAbs_qnt.iloc[h] = [errNetHr.iloc[h].quantile(0.25), errNetHr.iloc[h].quantile(0.5), errNetHr.iloc[h].quantile(0.75)]
    errScale.iloc[h] = [np.mean(errNetHr.iloc[h]) , np.mean(errNetHr.iloc[h]**2), np.sqrt(np.mean(errNetHr.iloc[h]**2))]

#%%
    
sns.set(style="whitegrid")


#plt.figure(figsize=(16,8))

plt.hist(errNet, bins=np.arange(10), density=True)

plt.xlabel('Net Error')
plt.ylabel('Density')
plt.title('Net EV Prediction Error')

#%%

sns.set(style="whitegrid")

font = {'family' : 'Times New Roman', 'size'   : 16}
plt.rc('font', **font)
plt.figure(figsize=(8,6))

#plt.hist(errNetAvg, bins=np.arange(10), density=True)
plt.plot(errNetHrAvg)

plt.xlabel('Net Error')
plt.xticks(np.arange(0,24,2))
plt.ylabel('Density')
plt.title('Average Gross Hourly EV Prediction Error')

#%% Results

sns.set(style="whitegrid")

font = {'family' : 'Times New Roman', 'size'   : 16}
plt.rc('font', **font)
plt.figure(figsize=(10,6))

for c in range(daysTrn):
    x = np.arange(24) + np.random.randn(24)/5
    plt.scatter(x, errNetHr[:,c])     

#%% Effective Sample Size
out_ess = pm.diagnostics.effective_n(EVtrace)
out_mu = EVtrace['mu'][5000:].mean(axis=0)
out_yExp = np.round(out_mu,0)
out_alpha = EVtrace['alpha'][5000:].mean(axis=0)

out_trace = pd.DataFrame.from_dict(list(EVtrace))
out_smry = pd.DataFrame(pm.summary(tracePoi))   

out_smry.to_excel('pool_Poi_5k_pt9.xlsx')
pm.save_trace(tracePoi, '/results/pool/pool_Poi_5k_pt9.trace')

#%%

#%% NB_Poi Quants

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(24)
q = 0.05;

qnt_TrnNB = pd.DataFrame(np.zeros((24,5)), columns=['Min','25pct', 'Med', '75pct','Max'])
qnt_TrnPoi = pd.DataFrame(np.zeros((24,5)), columns=['Min','25pct', 'Med', '75pct','Max'])
#dirPoi = 'results/1239704_10k_Poiss_NormP'; dirNB = 'results/1239578_10k_NB_NormP';

trace_NBmu = trace_both.where(trace_both.Dist =='NegBino')
trace_Poimu = trace_both.where(trace_both.Dist =='Poisson')

for h in np.arange(24): 
    
    NB_h = trace_NBmu.loc[trace_NBmu.Hour == h].mu
    Poi_h = trace_Poimu.loc[trace_Poimu.Hour == h].mu
    
    qnt_TrnNB.iloc[h] = [np.min(NB_h), NB_h.quantile(0.05), np.median(NB_h), 
                  NB_h.quantile(0.95), np.max(NB_h)]
    qnt_TrnPoi.iloc[h] = [np.min(Poi_h), Poi_h.quantile(0.05), np.median(Poi_h), 
                  Poi_h.quantile(0.95), np.max(Poi_h)]

#qnt_TrnNB = np.round(qnt_TrnNB,0)

#%% Fill Plot 

import random    
import seaborn as sns

daysInTest = list(set(dataTest.DayYr))
daysInTrn = list(set(dataTrn.DayYr))

daysIn = random.choices(daysInTest, k=1)

d_test = dataTest.loc[dataTest.DayYr == daysIn]
#% Day = 81, 188, 104, 61, 63, 65, 151

x = np.arange(24)

sns.set(style="whitegrid", font='Times New Roman', 
        font_scale=1.75)

fig, ax = plt.subplots(figsize=(12,8))

# Neg Bino Plot
ax.plot(x, qnt_TrnNB.Med, x, qnt_TrnNB['75pct'], color='black', lw=0.5)
ax.fill_between(x, qnt_TrnNB.Med, qnt_TrnNB['75pct'], where=qnt_TrnNB['75pct']>qnt_TrnNB.Med, 
                facecolor='blue', alpha=0.1)
ax.plot(x, qnt_TrnNB.Med, x, qnt_TrnNB['25pct'], color='black', lw=0.5)
ax.fill_between(x, qnt_TrnNB.Med, qnt_TrnNB['25pct'], where=qnt_TrnNB['25pct']<qnt_TrnNB.Med, 
                facecolor='blue', alpha=0.1)
ax.plot(x, qnt_TrnNB.Med, '--', c='blue', lw=2, label='NegBino')

## Poisson Plot
#ax.plot(x, qnt_TrnPoi.Med, x, qnt_TrnPoi['75pct'], color='black', lw=0.5)
#ax.fill_between(x, qnt_TrnPoi.Med, qnt_TrnPoi['75pct'], where=qnt_TrnPoi['75pct']>qnt_TrnPoi.Med, 
#                facecolor='orange', alpha=0.1)
#ax.plot(x, qnt_TrnPoi.Med, x, qnt_TrnPoi['25pct'], color='black', lw=0.5)
#ax.fill_between(x, qnt_TrnPoi.Med, qnt_TrnPoi['25pct'], where=qnt_TrnPoi['25pct']<qnt_TrnPoi.Med, 
#                facecolor='orange', alpha=0.1)
#ax.plot(x, qnt_TrnPoi.Med, '--', c='orange', lw=2, label='Poisson')

#plt.scatter(x,getTest.Int, marker='x', color='k', label='Actual')
plt.scatter(x,d_test.Connected, marker='x', color='k', label='Actual')

ax.set_title('NegBino')
ax.set_xticks(np.arange(0,26,2))
ax.set_xlabel('Hours')
ax.set_ylabel('EV Arrivals')
ax.legend()
print('daysIn ', daysIn)

#%% Traceplot

_ = pm.traceplot(trace[tunes:], 
                 var_names=['mu','alpha','hyper_mu_mu',
                           'hyper_mu_sd','hyper_alpha_mu',
                           'hyper_alpha_sd'])