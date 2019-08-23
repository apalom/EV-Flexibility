# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:19:25 2019

@author: Alex
In this section, we will look to answer:

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
import scipy.optimize as so
import theano.tensor as tt

data = pd.read_csv('data/hdc_wkdy.csv', index_col=0)
smpls = 100; tunes = 5; target = 0.9;
print('Params: samples = ', smpls, ' | tune = ', tunes, ' | target = ', target)
#%% Model Check II: Bayes Factor... Poisson vs. NegBino

with pm.Model() as model:

    # Index to true model
    prior_model_prob = 0.5
    #tau = pm.DiscreteUniform('tau', lower=0, upper=1)
    tau = pm.Bernoulli('tau', prior_model_prob)

    # Poisson parameters
    mu_p = pm.Uniform('mu_p', lower=0, upper=16)

    # Negative Binomial parameters
    #hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=100)
    #hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=200)

    #hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=100)
    #hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=200)

    #alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd)
    #mu_nb = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd)
    
    alpha = pm.Gamma('alpha', mu=4.50, sd=0.65)
    mu_nb = pm.Gamma('mu', mu=3.20, sd=1.25)

    y_like = pm.DensityDist('y_like',
             lambda value: pm.math.switch(tau,
                 pm.Poisson.dist(mu_p).logp(value),
                 pm.NegativeBinomial.dist(mu_nb, alpha).logp(value)
             ),
             observed=data.Connected.values)

    trace = pm.sample(smpls, tune=tunes, progressbar=False, nuts={"target_accept":target})

    #ess = pm.diagnostics.effective_n(trace)
    #print('- ESS: ', ess)

# Compute the Bayes factor
prob_pois = trace[int(0.25*smpls):]['tau'].mean()
print('Burn In prob_pois:' , prob_pois)
prob_pois = trace['tau'].mean()
print('prob_pois:' , prob_pois)
print(trace['tau'])
prob_nb = 1 - prob_pois
BF = (prob_nb/prob_pois)*(prior_model_prob/(1-prior_model_prob))
print("Bayes Factor: %s" % BF)
