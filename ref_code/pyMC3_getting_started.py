# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:53:00 2019

@author: Alex
"""

# Generating data
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

# Display generated data
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');

#%% Model Specification

import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)
    
print('--- Model Specified ---')
    
#%% Model Fitting

# Maximum a posteriori (MAP) estimate for a model, is the mode of the posterior 
# distribution and is generally found using numerical optimization methods
map_estimate = pm.find_MAP(model=basic_model)

map_estimate

# In summary, while PyMC3 provides the function find_MAP(), at this point mostly
# for historical reasons, this function is of little use in most scenarios. 
# If you want a point estimate you should get it from the posterior.
# In the next section we will see how to get a posterior using sampling methods.

print('--- Model Fit ---')

#%% Sampling Methods

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500)

print('--- Model Sampling ---')
#%% Posterior Analysis

# PyMC3 provides plotting and summarization functions for inspecting the sampling 
# output. A simple posterior plot can be created using traceplot.    
pm.traceplot(trace);

pm.summary(trace).round(2)