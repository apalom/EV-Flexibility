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

#%%

# Import Data
data = pd.read_csv('hdc_wkdy20.csv',  index_col='Idx');
data = data.head(30*24)

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

#%%
    
print('\n Running ', str(datetime.now()))

# Setup vars
smpls = 1000; tunes = 500; target = 0.80;

# Print Header
#print('hdc_wkdy20.csv | NB with Normal Prior')
print('hdc_wkdy20.csv | Poisson with Normal Prior')
print('Params: samples = ', smpls, ' | tune = ', tunes, ' | target = ', target, '\n')

#% Hierarchical Modeling
with pm.Model() as model:
    hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=20)
    hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=20)

    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=20)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=20)
    #hyper_mu_mu = pm.Normal('hyper_mu_mu', mu=2.965, sigma=3.307)

    alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd, shape=n_hrs)
    mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd, shape=n_hrs)

    #y_est = pm.Poisson('y_est', mu=mu[hrs_idx], observed=y_obs)
    #y_pred = pm.Poisson('y_pred', mu=mu[hrs_idx], shape=data.Hour.shape)

    y_est = pm.NegativeBinomial('y_est', 
                                mu=mu[hrs_idx], 
                                alpha=alpha[hrs_idx], 
                                observed=data['Connected'].values)
    
    y_pred = pm.NegativeBinomial('y_pred', 
                                 mu=mu[hrs_idx], 
                                 alpha=alpha[hrs_idx], 
                                 shape=data['Hour'].shape)
    
#%%

pm.model_to_graphviz(model)

with model:
    trace = pm.sample(smpls, tune=tunes, chains=4, progressbar=True, nuts={"target_accept": target})    

#trarr = pm.traceplot(trace[tunes:])

#pm.save_trace(trace, 'Pois_smpls' + str(int(smpls)) + '.trace')

ess = pm.diagnostics.effective_n(trace)

print('- ESS: ', ess)

out_yPred = trace.get_values('y_pred')[100::].ravel()
out_mu = trace['mu'][tunes:].mean(axis=0)
out_alpha = trace['alpha'][tunes:].mean(axis=0)

out_trace = pd.DataFrame.from_dict(list(trace))
#out_trace.to_csv('out_trace.csv')

out_smry = pd.DataFrame(pm.summary(trace))
#out_smry.to_csv('out_smry.csv')


#%%

_ = pm.traceplot(trace[tunes:], 
                 var_names=['mu','alpha','hyper_mu_mu',
                           'hyper_mu_sd','hyper_alpha_mu',
                           'hyper_alpha_sd'])

#%%

#log_connected = np.log(connected)
    
data = pd.read_csv('hdc_wkdy20.csv',  index_col='Idx');
data = data.head(30*24)
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
    y_like = pm.NegativeBinomial('y_like', 
                                 mu=mu[hrs_idx], 
                                 alpha=alpha[hrs_idx], 
                                 observed=connected)
    
#%%
    
with EVpooling:
    EVtrace = pm.sample(5000, tune=5000, cores=1, target_accept=.90)   
    

#%%
model_to_graphviz(EVpooling)    
    
#%%

plt.figure(figsize=(6,16))
pm.forestplot(EVtrace, var_names=['mu'])

#%%
out_mu = EVtrace['mu'][1000:].mean(axis=0)
out_muInt = np.round(out_mu,0)
    
    