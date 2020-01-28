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
per = "5min_1port";

for i in range(k):

    df_Train[i] = pd.read_excel("data/"+per+"/trn_test/trn"+str(i)+".xlsx")#.sample(10*288)    
    df_Test[i] = pd.read_excel("data/"+per+"/trn_test/test"+str(i)+".xlsx")#.sample(10*288)

#%% Bernoulli Model

def runModel(df_Train, df_Test, i, t, param, smpls, burns):

    dataTrn = df_Train[i]
    X = dataTrn[param].values     
    t_idx = dataTrn.Hour.astype(int).values
    
    dataTest = df_Test[i]
    testVals = pd.DataFrame(np.transpose(np.array([dataTest.Hour, dataTest[param].values])), columns=['hr','y'])
    testVals = testVals.groupby('hr').mean()
    testVals['int'] = np.round(testVals.y.values)
    
    # define bernoulli hierarchical model
    with pm.Model() as model:
    # define the hyperparameters
        mu = pm.Beta('mu', 2, 2)
        kappa = pm.Gamma('kappa', 1, 0.1)
        # define the prior
        theta = pm.Beta('theta', mu * kappa, (1 - mu) * kappa, shape=t)
        # define the likelihood
        y_lik = pm.Bernoulli('y_like', p=theta[t_idx], observed=X)
    
    #   Generate a MCMC chain
        trace = pm.sample(smpls, chains=4, tune=burns, cores=1)
        ppc = pm.sample_posterior_predictive(trace)
        
    out_smry = pd.DataFrame(pm.summary(trace))   
    
    ppcMean = np.array((t_idx, np.mean(ppc['y_like'], axis=0)))
    predVals = pd.DataFrame(np.transpose(ppcMean), columns=['hr', 'y'])
    predVals = predVals.groupby('hr').mean()
    predVals['int'] = np.round(predVals.y.values)
        
    err_y = np.round(SMAPE(testVals.y, predVals.y),4)
    err_int = np.round(SMAPE(testVals.int, predVals.int),4)
    print('\n Error: ', (err_y, err_int), '\n')
    
    return trace, ppc['y_like'], out_smry, [err_y, err_int]

def SMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

k = 5;
out_traces = {}; out_ppc = {}; out_smrys = {}; out_err = np.empty((k,2))

for i in range(5):
    # training data, testing data, folds, parameter, smpls , burnin
    out_traces[i], out_ppc[i], out_smrys[i], out_err[i] = runModel(X_Train, X_Test, i, 24, 'Connected', 500, 1000)
    
#%% Output Results

best = 3; 
np.savetxt("data/"+per+"/results/out_err.csv", err, delimiter=",")
np.savetxt("data/"+per+"/results/out_ppc.csv", out_ppc[best], delimiter=",")
out_smrys[best].to_excel("data/"+per+"/results/out_smry.xlsx")