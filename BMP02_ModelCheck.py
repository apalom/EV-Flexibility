# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:19:25 2019

@author: Alex
https://nbviewer.jupyter.org/github/markdregan/Bayesian-Modelling-in-Python/blob/master/Section%202.%20Model%20checking.ipynb

In this section, we will look at two techniques that aim to answer:

(1) Are the model and parameters estimated a good fit for the underlying data?
(2) Given two separate models, which is a better fit for the underlying data?
"""
# Import libraries
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import statsmodels.api as sm
import theano.tensor as tt

# (1) Model Check 1: Posterior predictive check
if __name__ == '__main__':
    
    with pm.Model() as model:
        mu = pm.Uniform('mu', lower=0, upper=100)
    
        y_est = pm.Poisson('y_est', mu=mu, observed=data.Connected.values)
        y_pred = pm.Poisson('y_pred', mu=mu)
        
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(10000, step, start=start, progressbar=True)
    
#%% Plot Posterior Predictive vs. Observed Values
        
# Conceptually, if the model is a good fit for the underlying data - then the 
# generated data should resemble the original observed data. PyMC provides a 
# convenient way to sample from the fitted model. 

# y_pred = pm.Poisson('y_pred', mu=mu)

# This is almost identical to y_est except we do not specify the observed data.
# PyMC considers this to be a stochastic node (as opposed to an observed node) 
# and as the MCMC sampler runs - it also samples data from y_est.
# We then plot y_pred below and compare it to the observed data y_est        
        
x_lim = 16
burnin = 2000

y_pred = trace[burnin:].get_values('y_pred')
mu_mean = trace[burnin:].get_values('mu').mean()

fig = plt.figure(figsize=(10,6))
fig.add_subplot(211)

_ = plt.hist(y_pred, range=[0, x_lim], density=True, bins=x_lim, color='red', edgecolor='white')   
_ = plt.ylabel('Frequency')
_ = plt.title('Posterior predictive distribution')

fig.add_subplot(212)

_ = plt.hist(data.Connected, range=[0, x_lim], density=True, bins=x_lim, edgecolor='white')   
_ = plt.xlabel('EVs Connected Over the Day')
_ = plt.ylabel('Frequency')
_ = plt.title('Distribution of observed data')

plt.tight_layout()

#%% Choosing the right distribution

# Perhaps the Poisson distribution is not suitable for this data. One alternative 
# option we have is the Negative Binomial distribution. This has very similar
# characteristics to the Poisson distribution except that it has two parameters 
# (μ and α) which enables it to vary its variance independently of its mean.
# Recall that the Poisson distribution has one parameter (μ) that represents 
# both its mean and its variance.

# Note... The Distribution of the Observed Data is "zero-inflated" as 
# compared to the Posterior Predictive from above.

fig = plt.figure(figsize=(10,5))
fig.add_subplot(211)
x_lim = 16
mu = [1, 3]
for i in np.arange(x_lim):
    plt.bar(i, stats.poisson.pmf(mu[0], i), fill=False, edgecolor='orange')
    plt.bar(i, stats.poisson.pmf(mu[1], i), fill=False, edgecolor='green')
    
_ = plt.xlabel('EVs Connected')
_ = plt.ylabel('Probability mass')
_ = plt.title('Poisson distribution')
_ = plt.legend(['$\lambda$ = %s' % mu[0],
                '$\lambda$ = %s' % mu[1]])

# Scipy takes parameters n & p, not mu & alpha
def get_n(mu, alpha):
    return 1. / alpha * mu

def get_p(mu, alpha):
    return get_n(mu, alpha) / (get_n(mu, alpha) + mu)

fig.add_subplot(212)

a = [2, 4]

for i in np.arange(x_lim):
    plt.bar(i, stats.nbinom.pmf(i, n=get_n(mu[0], a[0]), p=get_p(mu[0], a[0])), fill=False, edgecolor='orange')
    plt.bar(i, stats.nbinom.pmf(i, n=get_n(mu[1], a[1]), p=get_p(mu[1], a[1])), fill=False, edgecolor='green')

_ = plt.xlabel('EVs Connected')
_ = plt.ylabel('Probability mass')
_ = plt.title('Negative Binomial distribution')
_ = plt.legend(['$\\mu = %s, \/ \\beta = %s$' % (mu[0], a[0]),
                '$\\mu = %s, \/ \\beta = %s$' % (mu[1], a[1])])

plt.tight_layout()

#%% Estimate the parameters for a Negative Binomial 

if __name__ == '__main__':

    with pm.Model() as model:
        alpha = pm.Exponential('alpha', lam=.2)
        mu = pm.Uniform('mu', lower=0, upper=100)
        
        y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)
        y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=data.Connected.values)
        
        #start = pm.find_MAP()
        #step = pm.Metropolis()
        trace = pm.sample(10000, step, start=start, progressbar=True)