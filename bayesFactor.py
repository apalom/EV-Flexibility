# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:42:49 2019

@author: Alex
"""

import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import betaln
from scipy.stats import beta

plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

def beta_binom(prior, y):
    """
    Compute the marginal likelihood, analytically, for a beta-binomial model.

    prior : tuple
        tuple of alpha and beta parameter for the prior (beta distribution)
    y : array
        array with "1" and "0" corresponding to the success and fails respectively
    """
    alpha, beta = prior
    h = np.sum(y)
    n = len(y)
    p_y = np.exp(betaln(alpha + h, beta+n-h) - betaln(alpha, beta))
    return p_y

#%%
y = np.repeat([1, 0], [50, 50])  # 50 "heads" and 50 "tails"
priors = ((1, 1), (30, 30))

#%%
for a, b in priors:
    print('a,b: ',a,b)
    distri = beta(a, b)
    x = np.linspace(0, 1, 100)
    x_pdf = distri.pdf(x)
    plt.plot (x, x_pdf, label=r'$\alpha$ = {:d}, $\beta$ = {:d}'.format(a, b))
    plt.yticks([])
    plt.xlabel('$\\theta$')
    plt.legend()
    
BF = (beta_binom(priors[1], y) / beta_binom(priors[0], y))
print(round(BF))

#%%

n_chains = 1000

models = []
traces = []
for alpha, beta in priors:
    with pm.Model() as model:
        a = pm.Beta('a', alpha, beta)
        yl = pm.Bernoulli('yl', a, observed=y)
        trace = pm.sample(1000,
                          step=pm.SMC(),
                          random_seed=42)
        models.append(model)
        traces.append(trace)
        
#%%        
BF_smc = models[1].marginal_likelihood / models[0].marginal_likelihood
print((BF_smc))

#%% Read Trace from traceModelCheck.py

#path = 'results/1199637_traceCheck/out_hr8_trace.csv'

dataCheck = pd.read_csv('results/1199637_traceCheck/out_hr8_trace.csv', index_col=[0])

with pm.Model() as model:

    # Index to true model
    prior_model_prob = 0.5
    #tau = pm.DiscreteUniform('tau', lower=0, upper=1)
    tau = pm.Bernoulli('tau', prior_model_prob)
    trace = pm.sample(100, progressbar=True) 







