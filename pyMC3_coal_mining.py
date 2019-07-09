# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:07:04 2019

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))
import pandas as pd

# Generate Data
import pandas as pd
disaster_data = pd.Series([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                           3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                           2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
                           1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                           0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                           3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
years = np.arange(1851, 1962)

plt.plot(years, disaster_data, 'o', markersize=8);
plt.ylabel("Disaster count")
plt.xlabel("Year");

print('\n--- The Data ---')

#%% Model Specification with Discrete Variables

# Occurrences of disasters in the time series is thought to follow a Poisson 
# process with a large rate parameter in the early part of the time series, 
# and from one with a smaller rate in the later part. We are interested in 
# locating the change point in the series, which perhaps is related to changes 
# in mining safety regulations.

with pm.Model() as disaster_model:

    switchpoint = pm.DiscreteUniform('switchpoint', lower=years.min(), upper=years.max(), testval=1900)

    # Priors for pre- and post-switch rates number of disasters
    early_rate = pm.Exponential('early_rate', 1)
    late_rate = pm.Exponential('late_rate', 1)

    # Allocate appropriate Poisson rates to years before and after current
    rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)

    disasters = pm.Poisson('disasters', rate, observed=disaster_data)
    
# The major differences are the introduction of discrete variables with the 
# Poisson and discrete-uniform priors and the novel form of the deterministic
# random variable rate.
    
print('\n--- Model Spec ---')

#%% Model Sample

# Unfortunately because they are discrete variables and thus have no meaningful 
# gradient, we cannot use NUTS for sampling switchpoint or the missing disaster
# observations. Instead, we will sample using a Metroplis step method, which 
# implements adaptive Metropolis-Hastings, because it is designed to handle 
# discrete values. 

with disaster_model:
    trace = pm.sample(10000)
    
pm.traceplot(trace);

print('\n--- Model Sampling ---')

#%% Finding the Switch Point 

plt.figure(figsize=(10, 8))
plt.plot(years, disaster_data, '.')
plt.ylabel("Number of accidents", fontsize=16)
plt.xlabel("Year", fontsize=16)

plt.vlines(trace['switchpoint'].mean(), disaster_data.min(), disaster_data.max(), color='C1')
average_disasters = np.zeros_like(disaster_data, dtype='float')
for i, year in enumerate(years):
    idx = year < trace['switchpoint']
    average_disasters[i] = (trace['early_rate'][idx].sum() + trace['late_rate'][~idx].sum()) / (len(trace) * trace.nchains)

sp_hpd = pm.hpd(trace['switchpoint'])
plt.fill_betweenx(y=[disaster_data.min(), disaster_data.max()],
                  x1=sp_hpd[0], x2=sp_hpd[1], alpha=0.5, color='C1');
plt.plot(years, average_disasters,  'k--', lw=2);