# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:43:29 2019

@author: Alex
"""

def funcTrace24(path): # 'data/hr_day_cnctd.xlsx'
    
    import numpy as np    
    import pymc3 as pm
    import pandas as pd
    
    # When we want to understand the effect of more factors such as "day of week,"
    # "time of day," etc. We can use GLM (generalized linear models) to better
    # understand the effects of these factors.
    
    # Import Data
    data = pd.read_excel(path,  index_col='Index');
    
    #%% Houry NegativeBinomial Modeling
    # For each hour j and each EV connected i, we represent the model
    indiv_traces = {};
    
    # Convert categorical variables to integer
    hours = list(data.Hour)
    n_hours = len(hours)
    x_lim = 16
    
    print('---- Working -----')
    
    out_yPred = pd.DataFrame(np.zeros((x_lim,len(hours))), columns=list(hours))
    out_yObs = pd.DataFrame(np.zeros((x_lim,len(hours))), columns=list(hours))
    
    for h in hours:
        print('Hour: ', h)
        with pm.Model() as model:
            alpha = pm.Uniform('alpha', lower=0, upper=10)
            mu = pm.Uniform('mu', lower=0, upper=10)
    
            y_obs = data[data.Hour==h]['Connected'].values
            y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=y_obs)
    
            y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)
    
            trace = pm.sample(10000, progressbar=True)
    
            indiv_traces[h] = trace
    
        out_yPred.loc[:,h], _ = np.histogram(indiv_traces[h].get_values('y_pred'), bins=x_lim)
        out_yObs.loc[:,h], _ = np.histogram(data[data.Hour==h]['Connected'].values, bins=x_lim)
    
    # Export results
    out_yPred.to_csv('out_yPred.csv')
    out_yObs.to_csv('out_yObs.csv')

    return(out_yPred, out_yObs)