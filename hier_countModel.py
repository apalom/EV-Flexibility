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

#%% Read k-Fold Test-Train Data

df_Train = {}; df_Test = {};

df_Train[0] = pd.read_excel("data/1hr/trn_test/trn0.xlsx")
df_Test[0] = pd.read_excel("data/1hr/trn_test/test0.xlsx")
df_Train[1] = pd.read_excel("data/1hr/trn_test/trn1.xlsx")
df_Test[1] = pd.read_excel("data/1hr/trn_test/test1.xlsx")
df_Train[2] = pd.read_excel("data/1hr/trn_test/trn2.xlsx")
df_Test[2] = pd.read_excel("data/1hr/trn_test/test2.xlsx")
df_Train[3] = pd.read_excel("data/1hr/trn_test/trn3.xlsx")
df_Test[3] = pd.read_excel("data/1hr/trn_test/test3.xlsx")
df_Train[4] = pd.read_excel("data/1hr/trn_test/trn4.xlsx")
df_Test[4] = pd.read_excel("data/1hr/trn_test/test4.xlsx")

#%%  Import Data

#data = pd.read_excel('data/1hr/trn_test/trn0.xlsx')
#data = data.loc[data.DayCnt>0]#.sample(500)
##data = data.head(5000)
#X = data['Arrivals'].values 
#dataTest = pd.read_excel('data/1hr/trn_test/test0.xlsx')
#T = dataTest['Arrivals'].values

def runModel(df_Train, df_Test, i, param, smpls, burns):
    
    dataTrn = df_Train[i]
    X = dataTrn[param].values 
    
    dataTest = df_Test[i]    
    testVals = pd.DataFrame(np.transpose(np.array([dataTest.Hour, dataTest[param].values])), columns=['hr','y'])
    testVals = testVals.groupby('hr').mean()
    testVals['int'] = np.round(testVals.y.values)
    
    # Convert categorical variables to integer
    #hrs_idx = dataTrn['Hour'].values
    hrs_idx = dataTrn['Hour'].astype(int).values
    hrs = np.arange(96)
    n_hrs = len(hrs)
    
    # Setup Bayesian Hierarchical Model 
    with pm.Model() as countModel:
        
        # Define Hyper-Parameters
        hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=np.round(np.mean(X) + 3*np.std(X)))
        hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=np.round(np.mean(X) + 3*np.std(X))) 
        
        # Prior Definition
        mu = pm.Gamma('mu', mu=hyper_mu_mu, sigma=hyper_mu_sd, shape=n_hrs)    
        
        # Data Likelihood
        y_like = pm.Poisson('y_like', mu=mu[hrs_idx], observed=X)   
        
        # Data Prediction
    #    y_pred = pm.Poisson('y_pred', 
    #                        mu=mu[hrs_idx], 
    #                        shape=X.shape)
        
    #pm.model_to_graphviz(countModel)
    
    #% Hierarchical Model Inference    
    # Setup vars and print Header
    print('Poisson Likelihood')
    print('Params: samples = ', smpls, ' | tune = ', burns, '\n')
            
    with countModel:
        trace = pm.sample(smpls, chains=4, tune=burns, cores=1)#, NUTS={"target_accept": targetAcc})
        
        ppc = pm.sample_posterior_predictive(trace)
        #pm.traceplot(trace[burnin:], var_names=['mu'])                  
    
    out_smry = pd.DataFrame(pm.summary(trace))   
    
    ppcMean = np.array((hrs_idx, np.mean(ppc['y_like'], axis=0)))
    predVals = pd.DataFrame(np.transpose(ppcMean), columns=['hr', 'y'])
    predVals = predVals.groupby('hr').mean()
    predVals['int'] = np.round(predVals.y.values)
        
    err_y = np.round(SMAPE(testVals.y, predVals.y),4)
    err_int = np.round(SMAPE(testVals.int, predVals.int),4)
    print('\n Error: ', (err_y, err_int), '\n')
    
    return trace, ppc['y_like'], out_smry, (err_y, err_int)

def SMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

out_traces = {}; out_ppc = {}; out_smrys = {}; err = {};

for i in range(5):
    # training data, testing data, folds, parameter, smpls , burnin
    out_traces[i], out_ppc[i], out_smrys[i], err[i] = runModel(df_Train, df_Test, i, 'Arrivals', 1000, 4000)

#%%

resultHolder = {}
resultHolder['departures'] = (out_traces, out_ppc, out_smrys, err)
#%%

best = 1;
hrs_idx = df_Train[0]['Hour'].values;
df_ppc = pd.DataFrame(out_ppc[best][0::4]).T
df_ppc.insert(loc=0, column='Hr', value=hrs_idx)
df_ppc.to_csv("data/1hr/trn_test/hr1_arrival_ppc.csv")
df_smry = out_smrys[best];
df_smry.to_csv("data/1hr/trn_test/hr1_arrival_smry.csv")

#%% #% TracePlot

import arviz as az

plt.style.use('default')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

plt.figure(figsize=(10,4))
pm.traceplot(trace[0::4], var_names=['mu'])  

plt.figure(figsize=(10,4))
az.plot_posterior(trace[0::4])

plt.figure(figsize=(10,4))
az.plot_forest(trace[0::4], var_names=['mu'], combined=True)

#%% Train vs. Test Data

dataTest = pd.read_excel('data/1hr/trn_test/test4.xlsx')
T = dataTest['Arrivals'].values

s = 10000; t = len(T);
r = np.shape(ppc['y_like'])[0]; c = np.shape(ppc['y_like'])[1];
ppc_Smpl = pd.DataFrame(np.reshape(ppc['y_like'], (r*c,1)))
ppc_Smpl['Hour'] = np.tile(hrs_idx,r)
ppc_Smpl = ppc_Smpl.sample(s)

# Setup PPC
ppcTest = pd.DataFrame(np.zeros((s+t,3)), columns=['Hr', 'Departures', 'Src'])
ppcTest.Hr[0:s] = ppc_Smpl['Hour']
ppcTest.Departures[0:s] = ppc_Smpl[0]
ppcTest.Src[0:s] = 'PPC'
aggTrn = ppcTest[0:s]

# Setup test
ppcTest.Hr[s:s+t] = dataTest.Hour
ppcTest.Departures[s:s+t] = T
ppcTest.Src[s:s+t] = 'Test'
aggTest = ppcTest[s:s+t]

#% SNS Relationship Plot

import seaborn as sns

plt.style.use('default')
font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

g_Rel = sns.relplot(x='Hr', y='Departures', kind='line',
                 hue='Src', ci='sd', 
                 data=ppcTest)

g_Rel.fig.set_size_inches(12,6)
plt.xticks(np.arange(0,28,4))
plt.yticks(np.arange(0,18,2))

g_Data = pd.DataFrame(np.zeros((24,3)), columns=['Hr','Trn','Test'])

#%% Error Measure
import arviz as az
    
def SMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

font = {'family': 'Times New Roman', 'weight': 'light', 'size': 1}
plt.rc('font', **font)
g_Trn = sns.relplot(x='Hr', y='Departures', kind='line',
                 hue='Src', ci='sd', data=aggTrn)
g_Trn.fig.set_size_inches(0.1,0.1)

for ax in g_Trn.axes.flat:    
    for line in ax.lines:
        if len(line.get_xdata()) == 24:
            g_Data.Hr = line.get_xdata();
            g_Data.Trn = line.get_ydata();
    
g_Test = sns.relplot(x='Hr', y='Departures', kind='line',
                 hue='Src', ci='sd', data=aggTest)
g_Test.fig.set_size_inches(0.1,0.1)

for ax in g_Test.axes.flat:    
    for line in ax.lines:
        if len(line.get_xdata()) == 24:
            g_Data.Hr = line.get_xdata();
            g_Data.Test = line.get_ydata();

print('SMAPE: ', SMAPE(g_Data.Trn, g_Data.Test))

#print('r2 Trn: ', az.r2_score(X, np.array(ppc_Smpl[0].sample(len(X))))[0])
#print('r2 Test: ', az.r2_score(T, np.array(ppc_Smpl[0].sample(len(T))))[0])

