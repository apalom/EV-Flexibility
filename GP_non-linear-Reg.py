# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:20:57 2019

@author: Alex Palomino
https://pymc3-testing.readthedocs.io/en/rtd-docs/notebooks/GP-introduction.html
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cmap
cm = cmap.inferno

import numpy as np
import scipy as sp
import theano
import theano.tensor as tt
import theano.tensor.nlinalg
import sys
sys.path.insert(0, "../../..")
import pymc3 as pm
from pymc3 import *

#%% We generate 20 data points at random x values between 0 and 3.
# The true values of the hyperparameters are hardcoded in this 
# temporary model.

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

#%% GP Model Setup

Z = np.linspace(0,3,100)[:,None]

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

    y_obs = pm.gp.GP('y_obs', cov_func=f_cov, sigma=s2_n, observed={'X':X, 'Y':y})
    
#%% GP Model Inference
    
with model:
    trace = pm.sample(2000, cores=1)

pm.traceplot(trace[1000:], varnames=['l', 's2_f', 's2_n'],
             lines={"l": l_true,
                    "s2_f": s2_f_true,
                    "s2_n": s2_n_true});
    
#%% Generate data using throwaway PyMC3
    
np.random.seed(200)
n = 150
X = np.sort(40*np.random.rand(n))[:,None]

# define gp, true parameter values
with pm.Model() as model:
    l_per_true = 2
    cov_per = pm.gp.cov.Cosine(1, l_per_true)

    l_drift_true = 4
    cov_drift = pm.gp.cov.Matern52(1, l_drift_true)

    s2_p_true = 0.3
    s2_d_true = 1.5
    s2_w_true = 0.3

    periodic_cov = s2_p_true * cov_per
    drift_cov    = s2_d_true * cov_drift

    signal_cov   = periodic_cov + drift_cov
    noise_cov    = s2_w_true**2 * tt.eye(n)

K = theano.function([], signal_cov(X, X) + noise_cov)()
y = np.random.multivariate_normal(np.zeros(n), K)

#%% Qualitative Assessment of Periodicity

fig = plt.figure(figsize=(12,5)); ax = fig.add_subplot(111)
ax.plot(X, y, '--', color=cm(0.4))
ax.plot(X, y, 'o', color="k", ms=10);
ax.set_xlabel("x");
ax.set_ylabel("f(x)");

#%% infer the correct values of the hyperparameters

with pm.Model() as model:
    # prior for periodic lengthscale, or frequency
    l_per = pm.Uniform('l_per', lower=1e-5, upper=10)

    # prior for the drift lengthscale hyperparameter
    l_drift  = pm.Uniform('l_drift', lower=1e-5, upper=10)

    # uninformative prior on the periodic amplitude
    log_s2_p = pm.Uniform('log_s2_p', lower=-10, upper=5)
    s2_p = pm.Deterministic('s2_p', tt.exp(log_s2_p))

    # uninformative prior on the drift amplitude
    log_s2_d = pm.Uniform('log_s2_d', lower=-10, upper=5)
    s2_d = pm.Deterministic('s2_d', tt.exp(log_s2_d))

    # uninformative prior on the white noise variance
    log_s2_w = pm.Uniform('log_s2_w', lower=-10, upper=5)
    s2_w = pm.Deterministic('s2_w', tt.exp(log_s2_w))

    # the periodic "signal" covariance
    signal_cov = s2_p * pm.gp.cov.Cosine(1, l_per)

    # the "noise" covariance
    drift_cov  = s2_d * pm.gp.cov.Matern52(1, l_drift)

    y_obs = pm.gp.marginal_likelihood('y_obs', cov_func=signal_cov + drift_cov, sigma=s2_w, observed={'X':X, 'Y':y})
    
#%%