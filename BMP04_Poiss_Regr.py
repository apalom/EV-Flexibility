# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:30:51 2019

@author: Alex
https://github.com/markdregan/Bayesian-Modelling-in-Python
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import seaborn.apionly as sns
import statsmodels.api as sm
import theano.tensor as tt

from sklearn import preprocessing

# Import Data
dataTrain = pd.read_excel(r'data/dfFitsTrain_all.xlsx',  index_col='Index');
dataTest = pd.read_excel(r'data/dfFitsTest_all.xlsx',  index_col='Index');
          
        
#%% Define independent parameters

X = dataTrain[['Hour','DayWk','isWeekday']].values
_, num_X = X.shape

#%% Visualize distributions of independent variables

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False)

ax1.hist(X[:,0], bins=np.arange(0,25), density=True, edgecolor='white', linewidth=1.2, label='Hour')
ax1.set(xticks=np.arange(0,26,2), xlim=[-1, 25])
ax1.title.set_text('Hour')

ax2.hist(X[:,1], bins=np.arange(-1,8), density=True, color='green', edgecolor='white', linewidth=1.2, label='Hour')
ax2.set(xticks=np.arange(0,8,1), xlim=[-1, 8])
ax2.title.set_text('DayWk')

ax3.hist(X[:,2], bins=np.arange(0,3,1), density=True, color='orange', edgecolor='white', linewidth=1.2, label='Hour')
ax3.set(xticks=np.arange(0,2,1), xlim=[-1, 3])
ax3.title.set_text('isWeekday')

fig.tight_layout()

param_mean = {}; param_std = {};
param_mean['Hour'], param_mean['DayWk'], param_mean['isWeekday'] = np.mean(X, axis=0)
param_std['Hour'], param_std['DayWk'], param_std['isWeekday'] = np.std(X, axis=0)

#%% Link Functions
# If not modeling a continuous response variable from −∞ to ∞, one must 
# use a link function to transform your response range. For a Poisson distribution, 
# the canonical link function used is the log link.          

with pm.Model() as model:     
    
    # Priors for parameters
    intercept = pm.Normal('intercept', mu=0, sd=1)
    beta_Hour = pm.Uniform('beta_Hour', mu=0, sd=12)
    beta_DayWk = pm.Normal('beta_DayWk', mu=3.5, sd=4)
    beta_isWeekday = pm.Uniform('beta_isWeekday', mu=0, sd=1)
        
    mu = tt.exp(intercept 
                + beta_Hour*data.Hour
                + beta_isWeekday*data.isWeekday)
    
    y_est = pm.Poisson('y_est', mu=mu, observed=dataTrain['Connected'].values)
    
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(100000, step, start=start, progressbar=True)

_ = pm.traceplot(trace)