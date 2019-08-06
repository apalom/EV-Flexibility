# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:43:29 2019

@author: Alex
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import seaborn.apionly as sns

from IPython.display import Image
from sklearn import preprocessing

# When we want to understand the effect of more factors such as "day of week," 
# "time of day," etc. We can use GLM (generalized linear models) to better 
# understand the effects of these factors.

# Import Data
data = pd.read_excel('data/hr_day_cnctd.xlsx',  index_col='Index');

#%% Model Pooling Plot

fig = plt.figure(figsize=(12,3))

ax = data.groupby('Connected')['Hour'].size().plot(
        kind='bar', color='lightgrey', label='Zero Inflated')

ax = data.loc[data.Connected>0].groupby('Connected')['Hour'].size().plot(
        kind='bar', fill=False, edgecolor='green', label='Zeros Removed')

_ = ax.set_title('Number of Hours with x EVs Connected')
_ = ax.set_xlabel('EVs Connected')
_ = ax.set_ylabel('Hours With')

_ = plt.xticks(rotation=0)
_ = plt.legend()

#plt.savefig('observed.png')

#%% For each hour j and each EV connected i, we represent the model

# Convert categorical variables to integer
le = preprocessing.LabelEncoder()
data_idx = le.fit_transform(data.Hour) 
hours = le.classes_
n_hours = len(hours)

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
             var_names=['mu','alpha','hyper_mu_mu',
                       'hyper_mu_sd','hyper_alpha_mu',
                       'hyper_alpha_sd'])
    
pm.save_trace(hierarchical_trace, 'result/hierarch24.trace') 