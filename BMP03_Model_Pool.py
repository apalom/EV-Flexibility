# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:53:18 2019

@author: Alex
https://github.com/markdregan/Bayesian-Modelling-in-Python
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import seaborn.apionly as sns

from IPython.display import Image
from sklearn import preprocessing

# When we want to understand the effect of more factors such as "day of week," 
# "time of day," etc. We can use GLM (generalized linear models) to better 
# understand the effects of these factors.

#%% Import Data
data = pd.read_csv('data/hdc_wkdy20.csv',  index_col='Idx');
#dataNoZero = data.loc[data.Connected>0];
#dataTrain = pd.read_excel(r'data/dfFitsTrain_all.xlsx',  index_col='Index');
#dataTest = pd.read_excel(r'data/dfFitsTest_all.xlsx',  index_col='Index');
          
# Define independent parameters
#X = dataTrain[['Hour','DayWk','isWeekday']].values
#_, num_X = X.shape

#%% Model Pooling

fig = plt.figure(figsize=(12,3))

ax = data.groupby('Connected')['Hour'].size().plot(
        kind='bar', color='lightgrey', label='Zero Inflated')

ax = dataNoZero.groupby('Connected')['Hour'].size().plot(
        kind='bar', fill=False, edgecolor='green', label='Zeros Removed')

_ = ax.set_title('Number of Hours with x EVs Connected')
_ = ax.set_xlabel('EVs Connected')
_ = ax.set_ylabel('Hours With')

_ = plt.xticks(rotation=0)
_ = plt.legend()

#%% Calculate 24 hour normal distributions

hours = np.arange(24)
paramsNorm = pd.DataFrame(columns=['mu','stddev'])

for h in hours:
    dfTemp = data.loc[data.Hour == h]
    paramsNorm.loc[h] = [np.mean(dfTemp.Connected), np.std(dfTemp.Connected)]
    #paramsNorm.loc[h] = norm.fit(dfTemp.Connected)
    

#%% For each hour j and each EV connected i, we represent the model

indiv_traces = {}

# Convert categorical variables to integer
le = preprocessing.LabelEncoder()
data_idx = le.fit_transform(data.Hour) 
hours = le.classes_
n_hours = len(hours)

for h in [8]:
    print('Hour: ', h)
    with pm.Model() as model:
        alpha = pm.Uniform('alpha', lower=0, upper=20)
        mu = pm.Uniform('mu', lower=0, upper=20)
        
        y_obs = data[data.Hour==h]['Connected'].values
        
        y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=y_obs)

        y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)
        
        trace = pm.sample(25, progressbar=True)
        
        indiv_traces[h] = trace

#%% Plot NegBino Traces per Hour   
 
fig, axs = plt.subplots(n_hours, 2, figsize=(10, 48))
axs = axs.ravel()

colLeft = np.arange(0,48,2)
colRight = np.arange(1,48,2)

x_lim = 16

out_yPred = pd.DataFrame(np.zeros((x_lim,len(hours))), columns=list(hours)) 
out_yObs = pd.DataFrame(np.zeros((x_lim,len(hours))), columns=list(hours))       

for i, j, h in zip(colLeft, colRight, hours):
    axs[i].set_title('Observed Data Hr: %s' % h)
    axs[i].hist(data[data.Hour==h]['Connected'].values, range=[0, x_lim], 
       density=True, bins=x_lim, histtype='bar', rwidth=0.8, color='blue')
    axs[i].set_ylim([0, 1])

    axs[j].set_title('Posterior Predictive Distribution Hr: %s' % h)
    axs[j].hist(indiv_traces[h].get_values('y_pred'), range=[0, x_lim], 
       density=True, bins=x_lim, histtype='bar', rwidth=0.8, color='darkred')
    axs[j].set_ylim([0, 1])
    
    out_yPred.loc[:,h], _ = np.histogram(indiv_traces[h].get_values('y_pred'), bins=x_lim)
    out_yObs.loc[:,h], _ = np.histogram(data[data.Hour==h]['Connected'].values, bins=x_lim)

plt.tight_layout()

#for h in hours:
#    out_yPred.loc[:,h], _ = np.histogram(indiv_traces[h].get_values('y_pred'), bins=x_lim)
#    out_yObs.loc[:,h], _ = np.histogram(data[data.Hour==h]['Connected'].values, bins=x_lim)

# Export Data
out_yPred.to_csv('results/out_yPred.csv')
out_yObs.to_csv('results/out_yObs.csv')

#%% If we ombine the posterior predictive distributions across these models, 
# we would expect this to resemble the distribution of the overall dataset observed.

combined_y_pred = np.concatenate([v.get_values('y_pred') for k, v in indiv_traces.items()])

y_pred = trace.get_values('y_pred')

fig = plt.figure(figsize=(12,6))

fig.add_subplot(211)

_ = plt.hist(combined_y_pred, range=[0, x_lim], bins=x_lim, 
             rwidth=0.8, density=True, histtype='bar', color='darkred')   
#_ = plt.xlim(1, x_lim)
#_ = plt.ylim(0, 20000)
_ = plt.ylabel('Frequency')
_ = plt.title('Pooled: Posterior predictive distribution')

fig.add_subplot(212)

_ = plt.hist(data.Connected.values, range=[0, x_lim], bins=x_lim, 
             rwidth=0.8, density=True, histtype='bar', color='blue')  
#_ = plt.xlim(0, x_lim)
#_ = plt.xlabel('EVs Connected')
#_ = plt.ylim(0, 20)
_ = plt.ylabel('Frequency')
_ = plt.title('Pooled: Distribution of observed data')

plt.tight_layout()

#%% Hierarchal Model with Hyperparameters

with pm.Model() as model:  
    
    hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=15)
    hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=5)
    
    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=10)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=5)
    
    alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd, shape=n_hours)
    mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd, shape=n_hours)
    
    y_est = pm.NegativeBinomial('y_est', 
                                mu=mu[data_idx], 
                                alpha=alpha[data_idx], 
                                observed=data.Connected.values)
    
    y_pred = pm.NegativeBinomial('y_pred', 
                                 mu=mu[data_idx], 
                                 alpha=alpha[data_idx],
                                 shape=data.Hour.shape)

    hierarchical_trace = pm.sample(10000, progressbar=True)
    
_ = pm.traceplot(hierarchical_trace[6000:], 
             varnames=['mu','alpha','hyper_mu_mu',
                       'hyper_mu_sd','hyper_alpha_mu',
                       'hyper_alpha_sd'])

pm.save_trace(trace, 'data/hierarch24.trace') 

## later
#with model:
#   trace = pm.load_trace('data/hierarch24.trace') 
    
#%% Hourly Hierarchal Model with Hyperparameters

# Convert categorical variables to integer
h_trace24 = {};
outTrace_h24 = {};

hours = np.arange(24)
hours = [8]

n_hours = len(hours)

for h in hours:
    print('Hour: ', h)
    with pm.Model() as model:
        hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=15)
#        hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=5)
        hyper_alpha_mu = pm.Normal('hyper_alpha_mu', mu=paramsNorm.loc[h].stddev)
        
        hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=10)
#        hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=5)             
        hyper_mu_mu = pm.Normal('hyper_mu_mu', mu=paramsNorm.loc[h].mu)#, sigma=paramsNorm.loc[h].stddev)
        
        alpha1 = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd)
        mu1 = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd) 
        
        y_est = pm.NegativeBinomial('y_est', mu=mu1, alpha=alpha1, 
                                    observed=data.Connected.values)
        
        y_pred = pm.NegativeBinomial('y_pred', mu=mu1, alpha=alpha1)
                                     
    
        h_trace = pm.sample(1000, tune=50, progressbar=True)
        h_trace24[h] = h_trace;
        outTrace_h24[h] = list(h_trace24[h]);
        
#%%      
    _ = pm.traceplot(h_trace[8][5:], 
                 varnames=['mu','alpha','hyper_mu_mu',
                           'hyper_mu_sd','hyper_alpha_mu',
                           'hyper_alpha_sd'])
    
    pm.save_trace(trace, 'data/hierarch24.trace') 

    


