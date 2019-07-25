# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:53:18 2019

@author: Alex
https://github.com/markdregan/Bayesian-Modelling-in-Python
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
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

# Import Data
data = pd.read_excel('data/hr_day_cnctd.xlsx',  index_col='Index');
dataNoZero = data.loc[data.Connected>0];
dataTrain = pd.read_excel(r'data/dfFitsTrain_all.xlsx',  index_col='Index');
dataTest = pd.read_excel(r'data/dfFitsTest_all.xlsx',  index_col='Index');
          
# Define independent parameters
X = dataTrain[['Hour','DayWk','isWeekday']].values
_, num_X = X.shape

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

#%% For each hour j and each EV connected i, we represent the model

indiv_traces = {}

# Convert categorical variables to integer
le = preprocessing.LabelEncoder()
data_idx = le.fit_transform(data.Hour) 
hours = le.classes_
n_hours = len(hours)

for h in hours:
    print('Hour: ', h)
    with pm.Model() as model:
        alpha = pm.Uniform('alpha', lower=0, upper=10)
        mu = pm.Uniform('mu', lower=0, upper=10)
        
        y_obs = data[data.Hour==h]['Connected'].values
        y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=y_obs)

        y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)
        
        trace = pm.sample(10000, progressbar=True)
        
        indiv_traces[h] = trace

#%% Plot Traces per Hour
        
fig, axs = plt.subplots(n_hours,2, figsize=(24, 6))
axs = axs.ravel()

x_lim = 16

for i, j, h in zip(hours, hours, hours):
    axs[j].set_title('Observed: %s' % p)
    axs[j].hist(data[data.Hour==h]['Connected'].values,
        range=[0, x_lim], bins=x_lim, histtype='stepfilled')

for i, j, h in zip(hours, hours, hours):
    axs[j].set_title('Posterior predictive distribution: %s' % p)
    axs[j].hist(indiv_traces[hp].get_values('y_pred'), range=[0, x_lim], 
       bins=x_lim, histtype='stepfilled', color='lightgrey')

axs[4].set_xlabel('Response time (seconds)')
axs[5].set_xlabel('Response time (seconds)')

plt.tight_layout()