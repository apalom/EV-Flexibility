# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:15:44 2019

@author: Alex Palomino
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm

#dfDays_TrnVal = pd.read_csv('data/wkdy_Train_all.csv', index_col=[0])
#dfDays_TestVal = pd.read_csv('data/wkdy_Test_all.csv', index_col=[0])
#dfDays_Both = pd.concat([dfDays_TrnVal, dfDays_TestVal])
df = dfDays_Trn15Val

y_obs = df['Arrivals'].values
#y_obs = y_obs[0:48]
upprbnd = y_obs.mean() + 2 * y_obs.std()

# Convert categorical variables to integer
le = preprocessing.LabelEncoder()
hrs_idx = le.fit_transform(df['Hour'])
hrs = le.classes_
n_hrs = len(hrs)    

#%% Hierarchical Count Model
with pm.Model() as arrivalModel:
    
    # Hyper-Priors
    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=10)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=10) 
    
    # Priors   
    mu = pm.Gamma('mu', mu=hyper_mu_mu, 
                        sigma=hyper_mu_sd,
                        shape=n_hrs)    
    
    # Data Likelihood
    y_like = pm.Poisson('y_like', 
                       mu=mu[hrs_idx], 
                       observed=y_obs)    

pm.model_to_graphviz(arrivalModel)

#%% Hierarchical Energy Model
    
with pm.Model() as EVpooling:
    
    # Hyper-Priors    
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=upprbnd)
    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=upprbnd)
    
    hyper_sd_mu = pm.Uniform('hyper_sd_mu', lower=0, upper=upprbnd)
    hyper_sd_sd = pm.Uniform('hyper_sd_sd', lower=0, upper=upprbnd)
    
    # Priors
#    mu = pm.Normal('mu', mu=hyper_mu_mu, sigma=hyper_mu_sd,
#                        shape=n_hrs)    
#    sigma = pm.Normal('sigma', mu=hyper_sd_mu, sigma=hyper_sd_sd,
#                    shape=n_hrs) 
    mu = pm.Normal('mu', mu=hyper_mu_mu, sigma=hyper_mu_sd,
                    shape=n_hrs)
    
    sigma = pm.Normal('sigma', mu=hyper_sd_mu, sigma=hyper_sd_sd,
                    shape=n_hrs)
    
    # Data Likelihood
    y_like = pm.Normal('y_like', mu=mu[hrs_idx], sd=sigma[hrs_idx],
                       observed=y_obs)    
    
pm.model_to_graphviz(EVpooling)

#%% Hierarchical Model Inference

# Setup vars
smpls = 2500; tunes = 1000; 
    
# Print Header
print('\n Running ', str(datetime.now()))
print('Params: samples = ', smpls, ' | tune = ', tunes, '\n')
        
with arrivalModel:
    trace = pm.sample(smpls, chains=4, tune=tunes, cores=1, 
                      nuts_kwargs=dict(target_accept=0.90))
    ppc = pm.sample_posterior_predictive(trace)
    pm.traceplot(trace)      
    
out_smry = pd.DataFrame(pm.summary(trace))

#%%

for RV in EVpooling.basic_RVs:
    print(RV.name, RV.logp(EVpooling.test_point))
print(EVpooling.logp(EVpooling.test_point))
    
#%% Scatter Plot of Training Data Session Energy

import seaborn as sns
sns.set(style="whitegrid", font='Times New Roman', font_scale=1.75)
plt.figure(figsize=(16,8))

daysTot = len(set(df.DayCnt))
mean_kWh = df.Energy.groupby(df.Hour).mean()
sd_kWh = df.Energy.groupby(df.Hour).std()

for d in np.arange(daysTot): 
    plt.scatter(np.arange(0,24,0.25), df.Energy[4*24*d:(4*24*d)+4*24])
plt.plot(np.arange(0,24,0.25), mean_kWh,'w', lw=2)
    
plt.title('EV Charging Session Energy (Training Data)')
plt.xlabel('Hours (hr)')
plt.xticks(np.arange(0,26,2))
plt.ylabel('Energy (kWh)')

#%% Hourly Plot Histograms

df = dfDays_Both              
kBins = int(1 + 3.22*np.log(len(df))) #Sturge's Rule for Bin Count

fig, axs = plt.subplots(4, 6, figsize=(16,12), sharex=True, sharey=True) 
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
plt.rc('font', **font)

r,c = 0,0;
#Light Red = '#E3A79D' | Light Blue = #BB6FF
for hr in np.arange(24):          
    print('position', r, c)
    axs[r,c].hist(df.Energy.loc[df.Hour==hr].values, 
       edgecolor='white', color='skyblue', linewidth=0.5, 
       bins=np.arange(0,100,5), density=True) 
    axs[r,c].set_title('Hr: ' + str(hr))
    #axs[r,c].text(9, 0.35,  str(len(df_hr)) + ' samples')#, ha='center', va='center',)
    #axs[r,c].set_xlim(0,22)
    #axs[r,c].set_xticks(np.arange(0,22+4,4))
    
    # Subplot Spacing
    c += 1
    if c >= 6:
        r += 1; c = 0;
        if r >= 4:
            r=0;
  
fig.text(0.5, 0.0, 'Energy (kWh)', ha='center')
fig.text(-0.01, 0.5, 'Density', va='center', rotation='vertical')
fig.suptitle('Training & Testing Data Hourly Distribution Session Energy', y = 1.02)
plt.ylim(0,0.15)
plt.xlim(0,100)
plt.xticks(np.arange(0,105,15))

fig.tight_layout()
plt.show()

#%%

import theano.tensor as tt

class Mean(pm.gp.mean.Mean):

    def __init__(self, c=0):
        Mean.__init__(self)
        self.c = c

    def __call__(self, X):
        return tt.alloc(1.0, X.shape[0]) * self.c


