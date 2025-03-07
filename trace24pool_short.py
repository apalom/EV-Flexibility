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
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pymc3 as pm

#%%  Import Data

dataTrn = pd.read_csv('hdc_wkdy20.csv',  index_col='Idx');
dataTest = pd.read_csv('hdc_wkdy80.csv',  index_col='Idx');

# Convert categorical variables to integer
hrs_idx = dataTrn['Hour']
hrs = np.arange(24)
n_hrs = len(hrs)

dataHrly = pd.DataFrame(np.zeros((len(hrs),2)), columns=['mu', 'sd'])
for h in hrs:
    temp = dataTrn.loc[dataTrn.Hour == h]
    dataHrly.mu.at[h] = np.mean(temp.Connected)
    dataHrly.sd.at[h] = np.std(temp.Connected)

agg = [np.mean(dataTrn.Connected), np.std(dataTrn.Connected)]

#%% Load Data and Setup Hierarchical Model 
  
dataTrn = pd.read_csv('hdc_wkdy20.csv',  index_col='Idx');

cnctdTrn = dataTrn['Connected'].values
# Convert categorical variables to integer
hrs_idx = dataTrn['Hour']
hrs = np.arange(24)
n_hrs = len(hrs)
    
with pm.Model() as EVpooling:
    
    # Hyper-Priors
#    hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=5)
#    hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=15)

    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=10)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=10) 
    
    # Priors
#    alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, 
#                              sigma=hyper_alpha_sd, 
#                              shape=n_hrs)
    
    mu = pm.Gamma('mu', mu=hyper_mu_mu, 
                        sigma=hyper_mu_sd,
                        shape=n_hrs)    
    
    # Data Likelihood
    y_like = pm.Poisson('y_like', 
                       mu=mu[hrs_idx], 
                       observed=cnctdTrn)    
#    # Data Prediction
#    y_pred = pm.Poisson('y_pred', 
#                        mu=mu[hrs_idx], 
#                        shape=hrs_idx.shape)   
    
    #Data Likelihood
#    y_like = pm.NegativeBinomial('y_like', 
#                                 mu=mu[hrs_idx], 
#                                 alpha=alpha[hrs_idx], 
#                                 observed=cnctdTrn)
    # Data Prediction
#    y_pred = pm.NegativeBinomial('y_pred', 
#                                 mu=mu[hrs_idx], 
#                                 alpha=alpha[hrs_idx],
#                                 shape=hrs_idx.shape) 
#    
#    y_like = pm.Poisson('y_like', 
#                            mu=mu[hrs_idx],                                   
#                            observed=connected)
    

pm.model_to_graphviz(EVpooling)

#%% Hierarchical Model Inference

# Setup vars
smpls = 2500; tunes = 500; targetAcc = 0.90;    
    
# Print Header
print('\n Running ', str(datetime.now()))
#print('hdc_wkdy20.csv | NB with Uniform Hyper-Prior')
print('hdc_wkdy20.csv | Poisson with Uniform Hyper-Prior')
print('Params: samples = ', smpls, ' | tune = ', tunes, ' | target = ', targetAcc, '\n')
        
with EVpooling:
    trace = pm.sample(smpls, chains=4, tune=tunes, cores=1)#, NUTS={"target_accept": targetAcc})
    
    #ppc = pm.sample_posterior_predictive(trace)
    #pm.traceplot(trace)                  

out_smryPoi = pd.DataFrame(pm.summary(trace))  
#out_traceNB = pd.DataFrame.from_dict(list(trace)) 

#%% Compare Aggregate
ppc_Sample = pd.DataFrame(ppc['y_like'])
ppcVals = np.reshape(ppc_Sample.sample(10000).values, (10000*1248,1))
ppc_Sample = pd.DataFrame(ppc['y_like'].transpose())
ppc_Sample['Hr'] = hrs_idx 
ppcHrs = np.tile(hrs_idx,ppc_Sample.shape[1]-1)
ppc_Vals = pd.DataFrame(ppcVals, columns=['Connected'])
ppc_Vals['Hr'] = ppcHrs

#%%
nn = 15
errAgg = pd.DataFrame(np.zeros((nn, 6)), columns= ['nTrn', 'binTrn', 'nPPC', 'binPPC', 'nTest', 'binTest'])

import seaborn as sns
sns.set(style="whitegrid", font='Times New Roman', font_scale=1.75)
plt.figure(figsize=(16,8))

[errAgg.nTrn[0:nn-1], errAgg.binTrn, _] = plt.hist(dataTrn.Connected, 
                                                    bins=np.arange(nn), density=True, color='orange', label='Train')
[errAgg.nPPC[0:nn-1], errAgg.binPPC, _] = plt.hist(ppcVals, 
                                                    bins=np.arange(nn), density=True, color='blue', alpha=0.5, label='PPC')
[errAgg.nTest[0:nn-1], errAgg.binTest, _] = plt.hist(dataTest.Connected, 
                                                    bins=np.arange(nn), density=True, color='black', histtype='step', lw=3, label='Test')

plt.title('Poisson Aggregate Data Comparisons')
plt.xlabel('Connected EVs')
plt.ylabel('Density')
plt.legend()

#%% ppc Samples vs. test

ppc_Sample = pd.DataFrame(ppc['y_like']).transpose()
ppc_Sample['Hr'] = hrs_idx 

ppc_HistData = pd.DataFrame(np.zeros((24,13)))
for h in hrs:
    tempHr = ppc_Sample.loc[ppc_Sample.Hr==h].values[:,0:10000]
    [ppc_HistData.at[h], _] = np.histogram(tempHr, bins=np.arange(14), density=True)


test_HistData = pd.DataFrame(np.zeros((24,13)))  
for h in hrs:
    tempHr = dataTest.loc[dataTest.Hour==h].Connected
    [test_HistData.at[h], _] = np.histogram(tempHr, bins=np.arange(14), density=True)

#%%ppcVals = np.reshape(ppc_Sample.values, (ppc_Sample.shape[0]*ppc_Sample.shape[1],)) 

smpl = 25000; s=len(dataTest);
ppcTest = pd.DataFrame(np.zeros((smpl+s,3)), columns=['Connected', 'Hr', 'Src'])
ppcTest.Hr[0:smpl] = ppcHrs[0:smpl]
ppcTest.Connected[0:smpl] = ppcVals[0:smpl,0]
ppcTest.Src[0:smpl] = 'PPC'

ppcTest.Hr[smpl:smpl+s] = dataTest.Hour
ppcTest.Connected[smpl:smpl+s] = dataTest.Connected
ppcTest.Src[smpl:smpl+s] = 'Test'

#%% ppc Hourly Plot Histograms

import seaborn as sns
sns.set(style="whitegrid", font='Times New Roman', 
        font_scale=1.75)
              
fig, axs = plt.subplots(4, 6, figsize=(20,12), sharex=True, sharey=True) 
r,c = 0,0;

for h in hrs:     
    print('Hr: ', h)
    ppcSmpl = ppc_Vals.loc[ppc_Vals.Hr==h].sample(50)
    axs[r,c].hist(ppcSmpl.Connected, ec='white', fc='lightblue', 
                   bins=np.arange(16), density=True, label='Predicted')  
    print('Pred')
    testSmpl = dataTest.loc[dataTest.Hour==h]
    axs[r,c].hist(testSmpl.Connected, color='black', histtype='step', lw=2,
                   bins=np.arange(16), density=True, label='Testing')  
    print('Test')
    #axs[r,c].hist(get_nb_vals(mu, alpha, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='NB Dist')    
    #axs[r,c].hist(get_poiss_vals(mu, 1000), histtype='step', ec='black', bins=np.arange(16), density=True, lw=1.2, label='Poiss Dist')    
    #axs[r,c].hist(dctData[hr][dim].values, histtype='step', ec='blue', lw=1.2, bins=np.arange(16), density=True, label='Predicted') 
    axs[r,c].set_title('Hr: ' + str(h))
    
    # Subplot Spacing
    c += 1
    if c >= 6:
        r += 1;
        c = 0;
        if r >= 4:
            r=0;
    print(r,c)
  
fig.tight_layout()
fig.suptitle('Hourly Histogram Arrival Prediction', y = 1.02)
#xM, bS = int(np.max(dctData[hr][dim])), 4
xM, bS = 20, 4
plt.xlim(0,xM)
plt.xticks(np.arange(0,xM+bS,bS))
plt.ylim(0,0.4)
plt.legend()
plt.show()

#%% Calculate Hist MAPE, SMAPE

for h in hrs: 
    print(MAPE(ppc_HistData.iloc[h], test_HistData.iloc[h]))    
    
#%% ppcTest Violin Plot

import seaborn as sns
sns.set(style="whitegrid", font='Times New Roman', 
        font_scale=1.75)

plt.figure(figsize=(20,8))

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="Hr", y="Connected", data=ppcTest, 
               hue="Src", split=True, inner="box", scale='width', linewidth=1.25)

#plt.ylim(0, 20)
plt.title('Poisson Predictive Value Distributions')
plt.legend(title='')
plt.ylabel('EV Arrivals')

#%% Predictive Inference
dataTrn = pd.read_csv('hdc_wkdy80.csv',  index_col='Idx');

dataTrnS = dataTrn.sample(500)
obs_vals = dataTrnS.Connected
obs_hrsIdx = dataTrnS.Hour

with EVpooling:
    
    #Priors
    alpha_prd = pm.Gamma('alpha_prd', mu=hyper_alpha_mu, 
                                      sigma=hyper_alpha_sd, 
                                      shape=len(dataTrnS))
    
    mu1_prd = pm.Gamma('mu1_prd', mu=hyper_mu_mu, 
                                sigma=hyper_mu_sd,
                                shape=len(dataTrnS)) 
        
    y1_prd = pm.NegativeBinomial('y1_prd', mu=mu[obs_hrsIdx], 
                                         alpha=alpha[obs_hrsIdx], 
                                         observed=obs_vals)

with EVpooling:
    trace = pm.sample(2000, tune=1000, chains=2, cores=1)  
                     

#%% Load Trace
with EVpooling:
    tracePoi = pm.load_trace('results/pool/pool_Poi_5k_pt9.trace')

print('Poi Likelihood - 5000 sample - 5000 tune')

out_smryPoi = pd.DataFrame(pm.summary(tracePoi))  
out_tracePoi = pd.DataFrame.from_dict(list(tracePoi)) 
    
#%% Asking Questions of the Posterior
# pm.sample_ppc - https://docs.pymc.io/notebooks/posterior_predictive.html

level = 3
hr = 16

def hr_y_pred(hr):
    """Return posterior predictive for hr"""
    #ix = np.where(hrs_idx == hr)[0][:]
    #return trace['mu'][:, ix]
    ix = np.where(ppc_Vals.Hr == hr)[0][:]
    return ppc_Vals.Connected[ix]

def hr_pred_hist(hr):
    ix_check = hr_y_pred(hr) < level
    _ = plt.hist(hr_y_pred(hr)[ix_check], density=False,
                 bins=np.arange(16), fc='darkblue', ec='white', label='<= %s EV' %level)
    _ = plt.hist(hr_y_pred(hr)[~ix_check], density=False,
                 bins=np.arange(16), fc='lightblue', ec='white', label='> %s EV' %level)
    _ = plt.title('Posterior predictive \ndistribution for Hr: %s' %hr)
    _ = plt.xlabel('EV Arrivals')
    _ = plt.xticks(np.arange(0,18,2))
    _ = plt.ylabel('Frequency')
    _ = plt.legend()
    
def hr_pred_cdf(hr):
    x = np.linspace(0, 16, num=17)
    num_samples = float(len(hr_y_pred(hr)))
    prob_lt_x = [100*sum(hr_y_pred(hr) <= i)/num_samples for i in x]    
    _ = plt.plot(x, prob_lt_x, color='darkblue')
    _ = plt.fill_between(x, prob_lt_x, color='lightblue', alpha=0.3)
    y = prob_lt_x[level]
    _ = plt.scatter(level, y, s=100, c='darkblue')
    _ = plt.text(level+0.5, y-8, str(np.round(y,2)) + '%', fontsize=14)
    _ = plt.title('Probability of Arrivals at Hr: %s' % hr)
    _ = plt.xlabel('EV Arrivals')
    _ = plt.xticks(np.arange(0,18,2))
    _ = plt.ylabel('Cumulative Probability')
    _ = plt.ylim(ymin=0, ymax=100)
    _ = plt.xlim(xmin=0, xmax=16)

fig = plt.figure(figsize=(12,8))
_ = fig.add_subplot(221)
hr_pred_hist(hr)
_ = fig.add_subplot(222)
hr_pred_cdf(hr)
plt.tight_layout()

#%% SNS Relationship Plot

#plt.figure(figsize=(20,10))
import seaborn as sns

sns.set(style="whitegrid", font='Times New Roman', 
        font_scale=1.75)

g_test = sns.relplot(x='Hr', y='Connected', kind='line',
                 hue='Src', ci='sd', 
                 data=ppcTest)
g_test.fig.set_size_inches(16,8)

plt.xticks(np.arange(0,28,4))
plt.yticks(np.arange(0,16,2))

#%%
getRel = pd.DataFrame(np.zeros((24,5)), columns=['Hr','PPC', 'PPCint', 'Test', 'Testint'])
getRel.Hr = np.arange(24);
getRel.PPC = g_test.axes.flat[0].lines[0].get_ydata()
getRel.PPCint = np.round(getRel.PPC,0)
getRel.Test = g_test.axes.flat[1].lines[0].get_ydata()
getRel.Testint = np.round(getRel.Test,0)
print('MAPE = ', MAPE(getRel.Testint, getRel.PPCint))
print('SMAPE = ', SMAPE(getRel.Testint, getRel.PPCint))


#%%

x = np.linspace(0, 16, num=17)
within = pd.DataFrame(np.zeros((24,2)), columns=['one','two'])

for h in hrs:
    print(h)
    num_samples = float(520000.0)
    prob_lt_x = [100*sum(hr_y_pred(h) <= i)/num_samples for i in x]
    exp = np.int(np.mean(hr_y_pred(h)))
    val1 = prob_lt_x[exp+1] - prob_lt_x[exp-1]
    if exp == 0:
        val1 = prob_lt_x[exp+1];        
    within.one.at[h] = val1
    
    val2 = prob_lt_x[exp+2] - prob_lt_x[exp-2]
    if exp < 2:
        val2 = prob_lt_x[2];    
    within.two.at[h] = val2
    

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