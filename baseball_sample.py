# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:10:07 2019

@author: Alex
https://docs.pymc.io/notebooks/hierarchical_partial_pooling.html
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import theano.tensor as tt

dataBB = pd.read_csv(pm.get_data('efron-morris-75-data.tsv'), sep="\t")
at_bats, hits = dataBB[['At-Bats', 'Hits']].values.T

#%%
N = len(hits)

with pm.Model() as baseball_model:

    phi = pm.Uniform('phi', lower=0.0, upper=1.0)

    kappa_log = pm.Exponential('kappa_log', lam=1.5)
    kappa = pm.Deterministic('kappa', tt.exp(kappa_log))

    thetas = pm.Beta('thetas', alpha=phi*kappa, beta=(1.0-phi)*kappa, shape=N)
    y = pm.Binomial('y', n=at_bats, p=thetas, observed=hits)
    
#%%
    
with baseball_model:

    theta_new = pm.Beta('theta_new', alpha=phi*kappa, beta=(1.0-phi)*kappa)
    y_new = pm.Binomial('y_new', n=4, p=theta_new, observed=0)
        
#%%
    
with baseball_model:
    trace = pm.sample(2000, cores=1, tune=1000, chains=2,
                      target_accept=0.95)

pm.traceplot(trace, var_names=['phi', 'kappa']);

#%%

pm.model_to_graphviz(baseball_model)

#out_smry = pd.DataFrame(pm.summary(trace))

    #fig = plt.gcf()
    #fig.savefig("out_hr" + str(int(h)) + "_tracePlt" + ".png")