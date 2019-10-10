# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:38:43 2019

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))
import pandas as pd

# The Data
# Our data consist of 401 daily returns of the S&P 500 stock market index during the 2008 financial crisis.

returns = pd.read_csv(pm.get_data('SP500.csv'), parse_dates=True, index_col=0)

len(returns)

returns.plot(figsize=(10, 6))
plt.ylabel('daily returns in %');

print('\n--- The Data ---')
#%% Model Specification

with pm.Model() as sp500_model:
    nu = pm.Exponential('nu', 1/10., testval=5.)
    sigma = pm.Exponential('sigma', 1/0.02, testval=.1)

    s = pm.GaussianRandomWalk('s', sigma=sigma, shape=len(returns))
    volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s)**0.5)

    r = pm.StudentT('r', nu=nu, sigma=volatility_process, observed=returns['change'])

print('\n--- Model Specified ---')
    
#%% Model Fit

with sp500_model:
    trace = pm.sample(2000)

pm.traceplot(trace, varnames=['nu', 'sigma']);

print('\n--- Model Fit ---')

#%% Plot Distribution of Volatility Paths

fig, ax = plt.subplots(figsize=(15, 8))
returns.plot(ax=ax)
ax.plot(returns.index, 1/np.exp(trace['s',::5].T), 'C3', alpha=.03);
ax.set(title='volatility_process', xlabel='time', ylabel='volatility');
ax.legend(['S&P500', 'stochastic volatility process']);