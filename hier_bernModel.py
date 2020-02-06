# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:47:56 2020

@author: Alex Palomino
ref: http://www.cs.utah.edu/~fletcher/cs6190/lectures/BetaBernoulli.html
https://www.cs.ubc.ca/~schmidtm/Courses/540-W16/L19.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm
from pymc3 import model_to_graphviz

#%% Read k-Fold Test-Train Data

df_Train = {}; df_Val = {}; k = 5;
per = "5min_1port";

for i in range(k):

    df_Train[i] = pd.read_excel("data/"+per+"/trn_test/x_trn"+str(i)+".xlsx")#.sample(10*288)    
    df_Val[i] = pd.read_excel("data/"+per+"/trn_test/x_val"+str(i)+".xlsx")#.sample(10*288)
    print(i)

#%% Beta-Bernoulli Model

def runModel(df_Train, df_Val, i, t, param, smpls, burns):

    dataTrn = df_Train[i]
    X = dataTrn[param].values     
    t_idx = dataTrn.Hour.astype(int).values
    
    dataVal = df_Val[i]
    validate = pd.DataFrame(np.transpose(np.array([dataVal.Hour, dataVal[param].values])), columns=['hr','y'])
    validate = validate.groupby('hr').mean()
    validate['int'] = np.round(validate.y.values)
    
    # define bernoulli hierarchical model
    with pm.Model() as model:
    # define the hyperparameters
        mu = pm.Beta('mu', 2, 2)
        #mu = pm.Beta('mu', 0.5, 0.5)
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
    ppcStd = np.array((t_idx, np.std(ppc['y_like'], axis=0)))
    ppc_all = np.append(np.reshape(t_idx, (-1, 1)), ppc['y_like'].T, axis=1)
    predVals = pd.DataFrame(np.transpose(ppcMean), columns=['hr', 'y'])
    predVals = predVals.groupby('hr').mean()
    predVals['int'] = np.round(predVals.y.values)
                               
    # Calculate SMAPE Error                       
    err_y = np.round(SMAPE(validate.y, predVals.y),4)
    err_int = np.round(SMAPE(validate.int, predVals.int),4)
    print('\n Error: ', (err_y, err_int), '\n')
    
    return trace, ppc_all, out_smry, [err_y, err_int]

def SMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

k = 5;
out_traces = {}; out_ppcMean = {}; out_ppc = {}; 
out_smrys = {}; out_err = np.empty((k,2))

for i in range(5):
    # training data, testing data, folds, parameter, smpls , burnin
    out_traces[i], out_ppc[i], out_smrys[i], out_err[i] = runModel(X_Train, X_Test, i, 24, 'Connected', 500, 1000)
    
#%% Output Results

best = 1; 
np.savetxt("data/"+per+"/result/out_err.csv", out_err, delimiter=",")
np.savetxt("data/"+per+"/result/out_ppc.csv", out_ppc[best], delimiter=",")
out_smrys[best].to_excel("data/"+per+"/result/out_smry.xlsx")

#%% Test Values

#df_Test = pd.read_excel("data/"+per+"/trn_test/y_test.xlsx")#.sample(10*288)

daysIn = data_Test['Hour'].loc[data_Test['Hour'] == 0];

err_v = np.empty((len(daysIn),));
for d in range(len(daysIn)):
    
    actual = data_Test.Connected.loc[daysIn.index[d]:daysIn.index[d]+23].values;
    forecast = np.empty((24,));
    
    runs = 10; errs = np.empty((runs,));
    
    for i in range(runs):        
        for h in range(24):
            forecast[h] = stats.bernoulli.rvs(out_smrys[best]['mean'][h+2])
            
        errs[i] = np.sum(np.abs(forecast - actual))/24          
        
        #print(np.round(errs[i],2))
    err_v[d] = np.mean(errs)
    print('Err: ', np.round(err_v[d],2))

