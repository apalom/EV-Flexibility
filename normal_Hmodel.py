# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:11:14 2019

@author: Alex Palomino
"""

import pymc3 as pm
import theano.tensor as tt
import numpy as np

#X = np.tile(np.arange(0,10),100)
#y = np.random.normal(0, 2, 1000)

with pm.Model() as gp_fit:
    rho = pm.Gamma('rho', 1, 1)
    eta = pm.Gamma('eta', 1, 1)
    K = eta * pm.gp.cov.Matern32(1, rho)
    
    M = pm.gp.mean.Zero()
    sigma = pm.HalfCauchy('sigma', 2.5)
    
    gp = pm.gp.Latent(mean_func=M, cov_func=K)
    f = gp.prior("f", X)
    
    likelihood = pm.Normal()
    
    y_obs = pm.gp.gp.Marginal.marginal_likelihood('y_obs', mean_func=M, cov_func=K, 
                              sigma=sigma, observed={'X':X, 'Y':y})

    
with gp_fit:
    trace = pm.sample(2000, n_init = 20000)

pm.traceplot( trace[-1000:], varnames=['rho', 'sigma', 'eta'])

#%%
# https://docs.pymc.io/notebooks/GP-Marginal.html

import numpy as np
import pymc3 as pm

# A one dimensional column vector of inputs.
X = np.linspace(0, 1, 100)[:,None]
y = np.random.normal(0, 2.25, 100)

with pm.Model() as marginal_gp_model:
    # Specify the covariance function.
    cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.Marginal(cov_func=cov_func)

    # The scale of the white noise term can be provided,
    sigma = pm.HalfCauchy("sigma", beta=5)
    y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)
    
    trace = pm.sample(1000, chains=4, cores=1)

pm.plot_posterior(trace)

#%% vector of new X points we want to predict the function at
Xnew = np.linspace(0, 2, 100)[:, None]

with marginal_gp_model:
    f_star = gp.conditional("f_star", Xnew=Xnew)

    # or to predict the GP plus noise
    y_star = gp.conditional("y_star", Xnew=Xnew, pred_noise=True)
    
#%% The mean and full covariance
mu, cov = gp.predict(Xnew, point=trace[-1])

# The mean and variance (diagonal of the covariance)
mu, var = gp.predict(Xnew, point=trace[-1],  diag=True)

# With noise included
mu, var = gp.predict(Xnew, point=trace[-1],  diag=True, pred_noise=True)

#%% https://pymc3-testing.readthedocs.io/en/rtd-docs/notebooks/GP-introduction.html

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
cm = cmap.inferno

import numpy as np
import pandas as pd
import scipy as sp
import theano
import theano.tensor as tt
import theano.tensor.nlinalg
import sys
sys.path.insert(0, "../../..")
import pymc3 as pm

np.random.seed(20090425)
n = 20
X = np.sort(3*np.random.rand(n))[:,None]

with pm.Model() as model:
    # f(x)
    l_true = 0.3
    s2_f_true = 1.0
    cov = s2_f_true * pm.gp.cov.ExpQuad(1, l_true)

    # noise, epsilon
    s2_n_true = 0.1
    K_noise = s2_n_true**2 * tt.eye(n)
    K = cov(X) + K_noise

# evaluate the covariance with the given hyperparameters
K = theano.function([], cov(X) + K_noise)()

# generate fake data from GP with white noise (with variance sigma2)
y = np.random.multivariate_normal(np.zeros(n), K)

fig = plt.figure(figsize=(14,5)); ax = fig.add_subplot(111)
ax.plot(X, y, 'ok', ms=10);
ax.set_xlabel("x");
ax.set_ylabel("f(x)");

Z = np.linspace(0,3,0)[:,None]

#%%
with pm.Model() as model:
    # priors on the covariance function hyperparameters
    l = pm.Uniform('l', 0, 10)

    # uninformative prior on the function variance
    log_s2_f = pm.Uniform('log_s2_f', lower=-10, upper=5)
    s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))

    # uninformative prior on the noise variance
    log_s2_n = pm.Uniform('log_s2_n', lower=-10, upper=5)
    s2_n = pm.Deterministic('s2_n', tt.exp(log_s2_n))

    # covariance functions for the function f and the noise
    f_cov = s2_f * pm.gp.cov.ExpQuad(1, l)

    # Specify the GP.  The default mean function is `Zero`.
    gp = pm.gp.Marginal(cov_func=f_cov)

    yObs = gp.marginal_likelihood('yObs', X=X, y=y, noise=s2_n)
    
    trace = pm.sample(2000, chains=4, cores=1)

smry_out = pd.DataFrame(pm.summary(trace))    

#% Trace Plot
pm.traceplot(trace[500:], var_names=["l", "s2_f", "s2_n"]);

#%

gp_samples = pm.sample_posterior_predictive(trace[1000:], samples=500, model=model)

#%%
fig, ax = plt.subplots(figsize=(14,5))

[ax.plot(Z, x, color=cm(0.3), alpha=0.3) for x in gp_samples['yObs']]
# overlay the observed data
ax.plot(X, y, 'ok', ms=10);
ax.set_xlabel("x");
ax.set_ylabel("f(x)");
ax.set_title("Posterior predictive distribution");
