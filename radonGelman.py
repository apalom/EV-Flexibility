# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:04:30 2019

@author: Alex Palomino
https://docs.pymc.io/notebooks/multilevel_modeling.html
"""

import numpy as np
import pandas as pd
from pymc3 import __version__
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(__version__))

from pymc3 import get_data

# Import radon data
srrs2 = pd.read_csv(get_data('srrs2.dat'))
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state=='MN'].copy()

#%%
srrs_mn['fips'] = srrs_mn.stfips*1000 + srrs_mn.cntyfips
cty = pd.read_csv(get_data('cty.dat'))
cty_mn = cty[cty.st=='MN'].copy()
cty_mn['fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

#%%
srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
u = np.log(srrs_mn.Uppm)

n = len(srrs_mn)

#%%
srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))

#%%
county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
floor_measure = srrs_mn.floor.values

#%%
srrs_mn.activity.apply(lambda x: np.log(x+0.1)).hist(bins=25);

#%% Completely Pooled Model

from pymc3 import Model, sample, Normal, HalfCauchy, Uniform, model_to_graphviz

floor = srrs_mn.floor.values
log_radon = srrs_mn.log_radon.values

with Model() as pooled_model:

    beta = Normal('beta', 0, sigma=1e5, shape=2)
    sigma = HalfCauchy('sigma', 5)

    theta = beta[0] + beta[1]*floor

    y = Normal('y', theta, sigma=sigma, observed=log_radon)

model_to_graphviz(pooled_model)

with pooled_model:
    pooled_trace = sample(1000, cores=1, tune=1000)

#%%
    
b0, m0 = pooled_trace['beta'].mean(axis=0)


plt.scatter(srrs_mn.floor, np.log(srrs_mn.activity+0.1))
xvals = np.linspace(-0.2, 1.2)
plt.plot(xvals, m0*xvals+b0, 'r--');

#%% Unpooled Data

with Model() as unpooled_model:

    beta0 = Normal('beta0', 0, sigma=1e5, shape=counties)
    beta1 = Normal('beta1', 0, sigma=1e5)
    sigma = HalfCauchy('sigma', 5)

    theta = beta0[county] + beta1*floor

    y = Normal('y', theta, sigma=sigma, observed=log_radon)
    
model_to_graphviz(unpooled_model)

with unpooled_model:
    unpooled_trace = sample(1000, cores=1, tune=1000)

#%%

from pymc3 import forestplot

plt.figure(figsize=(6,14))
forestplot(unpooled_trace, var_names=['beta0'])

#%%

unpooled_estimates = pd.Series(unpooled_trace['beta0'].mean(axis=0), index=mn_counties)
unpooled_se = pd.Series(unpooled_trace['beta0'].std(axis=0), index=mn_counties)

order = unpooled_estimates.sort_values().index

plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
    plt.plot([i,i], [m-se, m+se], 'b-')
plt.xlim(-1,86); plt.ylim(-1,4)
plt.ylabel('Radon estimate');
plt.xlabel('Ordered county');

#%% Partial Pooling

with Model() as partial_pooling:

    # Priors
    mu_a = Normal('mu_a', mu=0., sigma=1e5)
    sigma_a = HalfCauchy('sigma_a', 5)

    # Random intercepts
    a = Normal('a', mu=mu_a, sigma=sigma_a, shape=counties)

    # Model error
    sigma_y = HalfCauchy('sigma_y',5)

    # Expected value
    y_hat = a[county]

    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sigma=sigma_y, observed=log_radon)

model_to_graphviz(partial_pooling)

with partial_pooling:
    partial_pooling_trace = sample(1000, cores=1, tune=1000)
    
#%% Varyging Intercept
    
with Model() as varying_intercept:

    # Priors
    mu_a = Normal('mu_a', mu=0., tau=0.0001)
    sigma_a = HalfCauchy('sigma_a', 5)


    # Random intercepts
    a = Normal('a', mu=mu_a, sigma=sigma_a, shape=counties)
    # Common slope
    b = Normal('b', mu=0., sigma=1e5)

    # Model error
    sd_y = HalfCauchy('sd_y', 5)

    # Expected value
    y_hat = a[county] + b * floor_measure

    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sigma=sd_y, observed=log_radon)

model_to_graphviz(varying_intercept)


with varying_intercept:
    varying_intercept_trace = sample(1000, cores=1, tune=1000)

#%% Varying Slope
    
with Model() as varying_slope:

    # Priors
    mu_b = Normal('mu_b', mu=0., sigma=1e5)
    sigma_b = HalfCauchy('sigma_b', 5)

    # Common intercepts
    a = Normal('a', mu=0., sigma=1e5)
    # Random slopes
    b = Normal('b', mu=mu_b, sigma=sigma_b, shape=counties)

    # Model error
    sigma_y = HalfCauchy('sigma_y',5)

    # Expected value
    y_hat = a + b[county] * floor_measure

    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sigma=sigma_y, observed=log_radon)

model_to_graphviz(varying_slope)

with varying_slope:
    varying_slope_trace = sample(1000, tune=1000, cores=1)
    
#%% Varying Slope and Intercept
    
with Model() as varying_intercept_slope:

    # Priors
    mu_a = Normal('mu_a', mu=0., sigma=1e5)
    sigma_a = HalfCauchy('sigma_a', 5)
    mu_b = Normal('mu_b', mu=0., sigma=1e5)
    sigma_b = HalfCauchy('sigma_b', 5)

    # Random intercepts
    a = Normal('a', mu=mu_a, sigma=sigma_a, shape=counties)
    # Random slopes
    b = Normal('b', mu=mu_b, sigma=sigma_b, shape=counties)

    # Model error
    sigma_y = Uniform('sigma_y', lower=0, upper=100)

    # Expected value
    y_hat = a[county] + b[county] * floor_measure

    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sigma=sigma_y, observed=log_radon)

model_to_graphviz(varying_intercept_slope)

#%% Group-level predictors

from pymc3 import Deterministic

with Model() as hierarchical_intercept:

    # Priors
    sigma_a = HalfCauchy('sigma_a', 5)

    # County uranium model for slope
    gamma_0 = Normal('gamma_0', mu=0., sigma=1e5)
    gamma_1 = Normal('gamma_1', mu=0., sigma=1e5)


    # Uranium model for intercept
    mu_a = gamma_0 + gamma_1*u
    
    # County variation not explained by uranium
    eps_a = Normal('eps_a', mu=0, sigma=sigma_a, shape=counties)
    a = Deterministic('a', mu_a + eps_a[county])

    # Common slope
    b = Normal('b', mu=0., sigma=1e5)

    # Model error
    sigma_y = Uniform('sigma_y', lower=0, upper=100)

    # Expected value
    y_hat = a + b * floor_measure

    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, sigma=sigma_y, observed=log_radon)

model_to_graphviz(hierarchical_intercept)

with hierarchical_intercept:
    hierarchical_intercept_trace = sample(1000, tune=1000, cores=1)