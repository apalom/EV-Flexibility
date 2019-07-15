# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:19:25 2019

@author: Alex
https://nbviewer.jupyter.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%202.%20Model%20checking.ipynb

In this section, we will look at two techniques that aim to answer:

(1) Are the model and parameters estimated a good fit for the underlying data?
(2) Given two separate models, which is a better fit for the underlying data?
"""
# Import libraries
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import statsmodels.api as sm
import theano.tensor as tt

# (1) Model Check 1: Posterior predictive check
if __name__ == '__main__':
    
    with pm.Model() as model:
        mu = pm.Uniform('mu', lower=0, upper=100)
    
        y_est = pm.Poisson('y_est', mu=mu, observed=data.Connected.values)
        y_pred = pm.Poisson('y_pred', mu=mu)
        
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(10000, step, start=start, progressbar=True)
    
#%% Plot Posterior Predictive vs. Observed Values
x_lim = 16
burnin = 5000

y_pred = trace[burnin:].get_values('y_pred')
mu_mean = trace[burnin:].get_values('mu').mean()

fig = plt.figure(figsize=(10,6))
fig.add_subplot(211)

_ = plt.hist(y_pred, range=[0, x_lim], bins=x_lim, histtype='stepfilled', color='red')   
_ = plt.xlim(1, x_lim)
_ = plt.ylabel('Frequency')
_ = plt.title('Posterior predictive distribution')

fig.add_subplot(212)

_ = plt.hist(data.Conneceed, range=[0, x_lim], bins=x_lim, histtype='stepfilled')
_ = plt.xlabel('Response time in seconds')
_ = plt.ylabel('Frequency')
_ = plt.title('Distribution of observed data')

plt.tight_layout()