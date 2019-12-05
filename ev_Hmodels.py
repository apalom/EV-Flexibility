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
dfDays_Trn15Val = pd.read_csv('data/dfDays_Trn15Val.csv', index_col=[0])

df = dfDays_Trn15Val

y_obs = df['Energy'].values[:,None]
#y_obs = y_obs[0:48]
#upprbnd = y_obs.mean() + 2 * y_obs.std()

# Convert categorical variables to integer
le = preprocessing.LabelEncoder()
hrs_idx = le.fit_transform(df['Hour'])[:,None]
hrs = le.classes_
n_hrs = len(hrs)    

#%% Hierarchical GP Energy Model

with pm.Model() as marginal_gp_model:
    # Specify the covariance function.
    input_dim = 1; T = 96; ls1 = 96; ls2 = 4;
    cov_func = ( pm.gp.cov.Periodic(input_dim, period=T, ls=ls1)
                + pm.gp.cov.ExpQuad(input_dim, ls=ls2) ) 

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.Marginal(cov_func=cov_func)

    # The scale of the white noise term can be provided,
    sigma = pm.HalfCauchy("sigma", beta=5)
    y_ = gp.marginal_likelihood("y", X=hrs_idx, y=y_obs, noise=sigma)
    
    trace = pm.sample(1000, chains=4, cores=1)

pm.plot_posterior(trace) 

#%% Hierarchical Poisson Count Model
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

#%% Calculate Parameter Values by MAP

# Zero Inflated
df = dfDays_Trn15Val.loc[dfDays_Trn15Val.Energy>1]
y_obs = df['Energy'].values[:,None]

# Convert categorical variables to integer
le = preprocessing.LabelEncoder()
hrs_idx = le.fit_transform(df['Hour'])[:,None]
hrs = le.classes_
n_hrs = len(hrs)   

#% Specify the covariance function.
input_dim = 1; T = 12; ls1 = 48; ls2 = 4; sd = 1;
#X=np.arange(0,10*96)[:,None];
X=hrs_idx[:10*96]
y=y_obs[:10*96].squeeze();

with pm.Model() as model:
    
    ℓ1 = pm.Uniform("ℓ1", lower=0, upper=ls1)
    τ1 = pm.Uniform("τ1", lower=0, upper=T)
    ℓ2 = pm.Uniform("ℓ2", lower=0, upper=ls2)
    
    cov = ( pm.gp.cov.Periodic(input_dim, period=τ1, ls=ℓ1)
            + pm.gp.cov.ExpQuad(input_dim, ls=ℓ2) ) 
    
    gp = pm.gp.Marginal(cov_func=cov)

    σ = pm.Normal("σ", mu=sd, sigma=2)
    y = gp.marginal_likelihood("y", X=np.arange(0,10*96)[:,None], 
                                    y=y_obs[:10*96].squeeze(), noise=σ)
    mp = pm.find_MAP()

# collect the results into a pandas dataframe to display
# "mp" stands for marginal posterior
pd.DataFrame({"Parameter": ["ℓ1", "τ1", "ℓ2", "σ"],
              "Value at MAP": [float(mp["ℓ1"]), float(mp["τ1"]), float(mp["ℓ2"]), float(mp["σ"])],
              "Init value": [ls1, T, ls2, sd]})
#%% Predict new values 
X_new = np.arange(0,2*96)[:,None]

# add the GP conditional to the model, given the new X values
with model:
    f_pred = gp.conditional("f_pred", X_new)

# To use the MAP values, you can just replace the trace with a length-1 list with `mp`
with model:
    pred_samples = pm.sample_posterior_predictive([mp], vars=[f_pred], samples=1000)
    
#%% Plot the results
    
fig = plt.figure(figsize=(16,8)); ax = fig.gca()

# plot the samples from the gp posterior with samples and shading
from pymc3.gp.util import plot_gp_dist
plot_gp_dist(ax, pred_samples["f_pred"], X_new);

# plot the data and the true latent function
y=y_obs[:10*96].squeeze();
plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Observed data");

# axis labels and title
plt.xlabel("Time (15min)"); plt.ylabel("Session Energy (kWh)");
plt.xticks(np.arange(0,len(X_new),24))
plt.ylim([0,20])
plt.title("Posterior distribution over $f(x)$ at the observed values"); plt.legend();