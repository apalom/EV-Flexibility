# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:23:55 2019

@author: Alex Palomino
"""

## Interactive magics

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import seaborn as sns
import patsy as pt
import pymc3 as pm

plt.rcParams['figure.figsize'] = 14, 6
np.random.seed(0)
print('Running Seaborn v{}'.format(sns.__version__))
print('Running on PyMC3 v{}'.format(pm.__version__))

#%% Load Functions

def strip_derived_rvs(rvs):
    '''Convenience fn: remove PyMC3-generated RVs from a list'''
    ret_rvs = []
    for rv in rvs:
        if not (re.search('_log',rv.name) or re.search('_interval',rv.name)):
            ret_rvs.append(rv)
    return ret_rvs


def plot_traces_pymc(trcs, varnames=None):
    ''' Convenience fn: plot traces with overlaid means and values '''

    nrows = len(trcs.varnames)
    if varnames is not None:
        nrows = len(varnames)

    ax = pm.traceplot(trcs, var_names=varnames, figsize=(12,nrows*1.4),
                      lines=tuple([(k, {}, v['mean'])
                                   for k, v in pm.summary(trcs, varnames=varnames).iterrows()]))

    for i, mn in enumerate(pm.summary(trcs, varnames=varnames)['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data',
                         xytext=(5,10), textcoords='offset points', rotation=90,
                         va='bottom', fontsize='large', color='#AA0022')
                         
#%% Generate Data
                         
# decide poisson theta values
theta_noalcohol_meds = 1    # no alcohol, took an antihist
theta_alcohol_meds = 3      # alcohol, took an antihist
theta_noalcohol_nomeds = 6  # no alcohol, no antihist
theta_alcohol_nomeds = 36   # alcohol, no antihist

# create samples
q = 100
df = pd.DataFrame({
        'nsneeze': np.concatenate((np.random.poisson(theta_noalcohol_meds, q),
                                   np.random.poisson(theta_alcohol_meds, q),
                                   np.random.poisson(theta_noalcohol_nomeds, q),
                                   np.random.poisson(theta_alcohol_nomeds, q))),
        'alcohol': np.concatenate((np.repeat(False, q),
                                   np.repeat(True, q),
                                   np.repeat(False, q),
                                   np.repeat(True, q))),
        'nomeds': np.concatenate((np.repeat(False, q),
                                      np.repeat(False, q),
                                      np.repeat(True, q),
                                      np.repeat(True, q)))})

#%% Preview Data      

# means of the various combinations (Poisson mean values)  
df.groupby(['alcohol','nomeds']).mean().unstack()        

# Data Category Plot
g = sns.catplot(x='nsneeze', row='nomeds', col='alcohol', data=df,
               kind='count', size=4, aspect=1.5)

#%% Poisson Regression Model

fml = 'nsneeze ~ alcohol + antihist + alcohol:antihist'  # full patsy formulation

fml = 'nsneeze ~ alcohol * nomeds'  # lazy, alternative patsy formulation

#%% Manual Method

#Create Design Matrices (using patsy as pt)
(mx_en, mx_ex) = pt.dmatrices(fml, df, return_type='dataframe', NA_action='raise')

pd.concat((mx_ex.head(3),mx_ex.tail(3)))

#%% Create Model

with pm.Model() as mdl_poiss:

    # define priors, weakly informative Normal
    b0 = pm.Normal('b0_intercept', mu=0, sigma=10)
    b1 = pm.Normal('b1_alcohol[T.True]', mu=0, sigma=10)
    b2 = pm.Normal('b2_nomeds[T.True]', mu=0, sigma=10)
    b3 = pm.Normal('b3_alcohol[T.True]:nomeds[T.True]', mu=0, sigma=10)

    # define linear model and exp link function
    theta = (b0 +
            b1 * mx_ex['alcohol[T.True]'] +
            b2 * mx_ex['nomeds[T.True]'] +
            b3 * mx_ex['alcohol[T.True]:nomeds[T.True]'])

    ## Define Poisson likelihood
    y = pm.Poisson('y', mu=np.exp(theta), observed=mx_en['nsneeze'].values)

#%% Sample Model
with mdl_poiss:
    trc_poiss = pm.sample(50, tune=10, cores=4)
    
#%% View Diagnostics
    
rvs_poiss = [rv.name for rv in strip_derived_rvs(mdl_poiss.unobserved_RVs)]
plot_traces_pymc(trc_poiss, varnames=rvs_poiss)

#%%
    