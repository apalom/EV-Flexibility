# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:10:56 2019

@author: Alex Palomino
http://blog.richardweiss.org/2017/10/18/pymc3-truncated-poisson.html
"""

import pandas as pd
import seaborn.apionly as sns
import pymc3 as pm
import matplotlib.pyplot as plt
from pymc3.distributions.dist_math import bound, logpow, factln
from pymc3.distributions import draw_values, generate_samples
import theano.tensor as tt
import numpy as np
import scipy.stats.distributions

#%% Generate Data

lams = np.asarray([0.79, 0.95])                    # The two lambdas
choices = np.random.choice(2, size=4000)          # Pick 4000 people, and give them groups
full_counts = np.random.poisson(lams[choices])    # Count their visits
truncated_counts = full_counts[full_counts > 0]   # Remove any counts that are zero
truncated_choices = choices[full_counts > 0]      # And also find the groups for those non-zero visitors
trunc_size = truncated_counts.size 
colors = sns.color_palette(n_colors=2)

#%% Setup/Plot Dummy Data

lam = 1
full_counts = np.random.poisson(lam, size=2000)
truncated_counts = full_counts[full_counts > 0]

sns.distplot(full_counts, bins=np.arange(10), kde=False, norm_hist=True)
sns.distplot(truncated_counts, bins=np.arange(10), kde=False, norm_hist=True)

#%% Full Counts

with pm.Model():
    lam = pm.HalfNormal('lam', 10)
    pm.Poisson('obs', mu=lam, observed=full_counts)
    
    trace = pm.sample(2500, cores=1)
    pm.traceplot(trace)
    
plt.figure()
sns.distplot(trace.lam)
plt.axvline(1)

#%% Truncated Counts

with pm.Model():
    lam = pm.HalfNormal('lam', 10)
    pm.Poisson('obs', mu=lam, observed=truncated_counts)
    
    trace = pm.sample(2500, cores=1)
    pm.traceplot(trace)
    
plt.figure()
sns.distplot(trace.lam)
plt.axvline()

#%% Define ZTP

class ZTP(pm.Discrete):
    def __init__(self, mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = tt.minimum(tt.floor(mu).astype('int32'), 1)
        self.mu = mu = tt.as_tensor_variable(mu)

    def zpt_cdf(self, mu, size=None):
        mu = np.asarray(mu)
        dist = scipy.stats.distributions.poisson(mu)
        
        lower_cdf = dist.cdf(0)
        upper_cdf = 1
        nrm = upper_cdf - lower_cdf
        sample = np.random.random(size=size) * nrm + lower_cdf

        return dist.ppf(sample)
        
    def random(self, point=None, size=None, repeat=None):
        mu = draw_values([self.mu], point=point)
        return generate_samples(self.zpt_cdf, mu,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        #              mu^k
        #     PDF = ------------
        #            k! (e^mu - 1)
        # log(PDF) = log(mu^k) - (log(k!) + log(e^mu - 1))
        #
        # See https://en.wikipedia.org/wiki/Zero-truncated_Poisson_distribution
        p = logpow(mu, value) - (factln(value) + pm.math.log(pm.math.exp(mu)-1))
        log_prob = bound(
            p,
            mu >= 0, value >= 0)
        # Return zero when mu and value are both zero
        return tt.switch(1 * tt.eq(mu, 0) * tt.eq(value, 0),
                         0, log_prob)

#%% ZTP Model

with pm.Model():
    lam = pm.HalfNormal('lam', 10, shape=1)
    ZTP('obs', mu=lam[np.zeros_like(truncated_counts)], observed=truncated_counts)
    
    trace = pm.sample(2500, cores=1)
    pm.traceplot(trace)
    
plt.figure()
sns.distplot(trace.lam)
plt.axvline(1)

#%% Setup New Dummy Data

rng = np.random.RandomState(4353420)

lams = np.asarray([0.79, 0.95])
choices = rng.choice(2, size=4000)
full_counts = rng.poisson(lams[choices])
truncated_counts = full_counts[full_counts > 0]
truncated_choices = choices[full_counts > 0]
trunc_size = truncated_counts.size
colors = sns.color_palette(n_colors=2)

#%% Clever ZPT Model

with pm.Model():
    lam = pm.HalfNormal('lam', 10, shape=2)
    ZTP('obs', mu=lam[truncated_choices], observed=truncated_counts)
    
    trace = pm.sample(2500, cores=1)
    ppc = pm.sample_ppc(trace)
    pm.traceplot(trace)

#%%  
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,4))
pm.plot_posterior(trace, color='lightgrey', ax=axs[0, :])
axs[0, 0].axvline(lams[0], color=colors[0], linewidth=2)
axs[0, 1].axvline(lams[1], color=colors[1], linewidth=2)
dist_1 = ppc['obs'].squeeze()[:, truncated_choices == 0].ravel()
dist_2 = ppc['obs'].squeeze()[:, truncated_choices == 1].ravel()
true_1 = truncated_counts[truncated_choices == 0]
true_2 = truncated_counts[truncated_choices == 1]

#%%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,4))
sns.distplot(dist_1, ax=axs[1, 0], norm_hist=True, kde=False, bins=np.arange(10), label='PPC', color=colors[0])
sns.distplot(true_1, ax=axs[1, 0], norm_hist=True, kde=False, bins=np.arange(10), label='Actual', color='grey')
sns.distplot(dist_2, ax=axs[1, 1], norm_hist=True, kde=False, bins=np.arange(10), label='PPC', color=colors[1])
sns.distplot(true_2, ax=axs[1, 1], norm_hist=True, kde=False, bins=np.arange(10), label='Actual', color='grey')
axs[1, 1].legend()
axs[1, 0].legend()

pm.summary(trace)