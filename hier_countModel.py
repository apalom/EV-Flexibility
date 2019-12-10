# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:16:49 2019

@author: Alex
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm

#%%  Import Data

X = df_Train[0]['Arrivals'].values

# Convert categorical variables to integer
hrs_idx = df_Train[0]['Hour'].astype(int)
hrs = np.arange(96)
n_hrs = len(hrs)

# Setup Bayesian Hierarchical Model 
with pm.Model() as countModel:
    
    # Define Hyper-Parameters
    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=np.round(np.mean(X) + 3*np.std(X)))
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=np.round(np.mean(X) + 3*np.std(X))) 
    
    # Prior Definition
    mu = pm.Gamma('mu', mu=hyper_mu_mu, 
                        sigma=hyper_mu_sd,
                        shape=n_hrs)    
    
    # Data Likelihood
    y_like = pm.Poisson('y_like', 
                       mu=mu[hrs_idx], 
                       observed=X)   
    
    # Data Prediction
#    y_pred = pm.Poisson('y_pred', 
#                        mu=mu[hrs_idx], 
#                        shape=X.shape)
    
pm.model_to_graphviz(countModel)
    
#%% Hierarchical Model Inference

# Setup vars
smpls = 2500; burnin = 25000;

# Print Header
print('Poisson Likelihood')
print('Params: samples = ', smpls, ' | tune = ', burnin, '\n')
        
with countModel:
    trace = pm.sample(smpls, chains=4, tune=burnin, cores=1)#, NUTS={"target_accept": targetAcc})
    
    #ppc = pm.sample_posterior_predictive(trace)
    #pm.traceplot(trace[burnin:], var_names=['mu'])                  

out_smryPoi = pd.DataFrame(pm.summary(trace))  

#trace_Stn = trace[burnin:];
#trace_Smpls = trace_Stn[0::10]