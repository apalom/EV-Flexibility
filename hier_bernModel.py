# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:47:56 2020

@author: Alex Palomino
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm

#%% Read k-Fold Test-Train Data

df_Train = {}; df_Test = {}; k = 5;
per = "5min_1chgr";

for i in range(k):

    df_Train[i] = pd.read_excel("data/"+per+"/trn_test/trn"+str(i)+".xlsx")    
    df_Test[i] = pd.read_excel("data/"+per+"/trn_test/test"+str(i)+".xlsx")
    
#%% Bernoulli Model

X = df_Train[0].Connected
y = df_Test[0].Connected

t_idx = df_Train[0].Hour.astype(int).values
N = len(t_idx)

#%%
with pm.Model() as model:
# define the hyperparameters
    mu = pm.Beta('mu', 2, 2)
    kappa = pm.Gamma('kappa', 1, 0.1)
    # define the prior
    theta = pm.Beta('theta', mu * kappa, (1 - mu) * kappa, shape=len(N))
    # define the likelihood
    y = pm.Bernoulli('y', p=theta[t_idx], observed=X)

#   Generate a MCMC chain
    trace = pm.sample(500, progressbar=True)
        
