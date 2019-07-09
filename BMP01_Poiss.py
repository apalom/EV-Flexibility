# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:40:43 2019

@author: Alex
https://github.com/markdregan/Bayesian-Modelling-in-Python
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm

# Import Data
data = pd.read_excel('data/hr_day_cnctd.xlsx',  index_col='Index');

#%% Frequentists method of estimating μ

y_obs = data.Connected;

def poisson_logprob(mu, sign=-1):
    return np.sum(sign*stats.poisson.logpmf(y_obs, mu=mu))

freq_results = opt.minimize_scalar(poisson_logprob)
print("The estimated value of mu is: %.3f" % freq_results['x'])

freq_mu = freq_results['x'];

# The frequentist optimization technique doesn't provide any measure of uncertainty
# it just returns a point value. 

#%% Plot Optimization
# This plot illustrates the function that we are optimizing. At each value of μ,
# the plot shows the log probability at μ given the data and the model. The optimizer 
# works in a hill climbing fashion - starting at a random point on the curve and 
# incrementally climbing until it cannot get to a higher point.

x = np.linspace(1, 5)
y_min = np.min([poisson_logprob(i, sign=1) for i in x])
y_max = np.max([poisson_logprob(i, sign=1) for i in x])

fig = plt.figure(figsize=(6,4))
_ = plt.plot(x, [poisson_logprob(i, sign=1) for i in x])
_ = plt.fill_between(x, [poisson_logprob(i, sign=1) for i in x], 
                     y_min, alpha=0.3)
_ = plt.title('Optimization of $\mu$')
_ = plt.xlabel('$\mu$')
_ = plt.ylabel('Log probability of $\mu$ given data')
_ = plt.vlines(freq_results['x'], y_max, y_min, colors='red', linestyles='dashed')
_ = plt.scatter(freq_results['x'], y_max, s=110, c='red', zorder=3)
_ = plt.ylim(ymin=y_min, ymax=0)
_ = plt.xlim(xmin=1, xmax=2*freq_mu)

#%% Plot 

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
x_lim = np.max(data.Connected);
mu = np.int(np.round(freq_results['x']))
for i in np.arange(x_lim):
    plt.bar(i, stats.poisson.pmf(mu, i), color='blue')
    
_ = ax.set_xlim(0, x_lim)
#_ = ax.set_ylim(0, 0.2)
_ = ax.set_xlabel('EVs Connected')
_ = ax.set_ylabel('Probability mass')
_ = ax.set_title('Frequentist Estimated Poisson distribution for EVs Connected')
_ = plt.legend(['$\lambda$ = %s' % mu])

#%% Bayesian Approach (using MCMC sampler)

print('Running on PyMC3 v{}'.format(pm.__version__))

with pm.Model() as model:
    mu = pm.Uniform('mu', lower=0, upper=60)
    likelihood = pm.Poisson('likelihood', mu=mu, observed=y_obs)
    
    start = pm.find_MAP()
    step = pm.Metropolis()

with model:    s
    trace = pm.sample(10000, step, start=start, progressbar=True)
    
#% Optimal Mu
    
#pm.traceplot(trace, var_names=['mu'], lines={'mu': freq_results['x']})
pm.traceplot(trace)

print('\n--- Optimal Model Parameters ---')

#%% Discarding early samples (burnin)

fig = plt.figure(figsize=(10,4))
plt.subplot(121)
_ = plt.title('Burnin trace')
_ = plt.ylim(freq_mu - 0.2, freq_mu + 0.2) 
_ = plt.plot(trace.get_values('mu')[:1000])

fig = plt.subplot(122)
_ = plt.title('Full trace')
_ = plt.ylim(freq_mu - 0.2, freq_mu + 0.2) 
_ = plt.plot(trace.get_values('mu'))

#%% Autocorrelation Test Plot
# A measure of correlation between successive samples in the MCMC sampling chain.
# When samples have low correlation with each other, they are adding more "
# information" to the estimate of your parameter value than samples that are 
# highly correlated.

# Visually, you are looking for an autocorrelation plot that tapers off to zero 
# relatively quickly and then oscilates above and below zero correlation. If 
# your autocorrelation plot does not taper off - it is generally a sign of poor 
# mixing and you should revisit your model selection (eg. likelihood) and 
# sampling methods (eg. Metropolis).

_ = pm.autocorrplot(trace[:2000], var_names=['mu'])