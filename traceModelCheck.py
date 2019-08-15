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
import scipy.optimize as so
import theano.tensor as tt

data = pd.read_csv('data/hdc_wkdy_TRAIN20.csv', index_col=0)
smpls = 10000
        
#%% Model Check II: Bayes Factor... Poisson vs. NegBino

with pm.Model() as model:
    
    # Index to true model
    prior_model_prob = 0.5
    #tau = pm.DiscreteUniform('tau', lower=0, upper=1)
    tau = pm.Bernoulli('tau', prior_model_prob)
    
    # Poisson parameters
    mu_p = pm.Uniform('mu_p', lower=0, upper=16)

    # Negative Binomial parameters
    hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=10)
    hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=25)
        
    hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=10)
    hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=25)
        
    alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd)
    mu_nb = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd)

    y_like = pm.DensityDist('y_like',
             lambda value: pm.math.switch(tau, 
                 pm.Poisson.dist(mu_p).logp(value),
                 pm.NegativeBinomial.dist(mu_nb, alpha).logp(value)
             ),
             observed=data.Connected.values)
    
    #start = pm.find_MAP()
    #step1 = pm.Metropolis([mu_p, alpha, mu_nb])
    #step2 = pm.ElemwiseCategorical(vars=[tau], values=[0,1])
    #trace = pm.sample(10000, step=[step1, step2], start=start)
    trace = pm.sample(1000, progressbar=True)
    
    ess = pm.diagnostics.effective_n(trace)

    print('- ESS: ', ess)

# Compute the Bayes factor
prob_pois = trace[0.25*smpls:]['tau'].mean()
prob_nb = 1 - prob_pois
BF = (prob_nb/prob_pois)*(prior_model_prob/(1-prior_model_prob))
print("Bayes Factor: %s" % BF)

