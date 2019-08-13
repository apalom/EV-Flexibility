#~/VENV3.6.3/bin/

# import sys
# sys.path.append('/uufs/chpc.utah.edu/sys/installdir/python/3.6.3/lib/python3.6/site-packages/')
# import os
# sys.path.append(os.getcwd())

from datetime import datetime
import numpy as np
import scipy.stats as stats
import pymc3 as pm
import pandas as pd

# When we want to understand the effect of more factors such as "day of week,"
# "time of day," etc. We can use GLM (generalized linear models) to better
# understand the effects of these factors.
print('Running ', str(datetime.now()))

# Import Data
data = pd.read_csv('data/hdc_wkdy_TRAIN.csv',  index_col='Idx');

#%% Houry NegativeBinomial Modelings
# For each hour j and each EV connected i, we represent the model
indiv_traces = {};

# Convert categorical variables to integer
hours = np.arange(0,24)
n_hours = len(hours)
bins16 = np.arange(0,16)

out_yPred = {};
out_yObs = {};
trace24 = {};

#writer = pd.ExcelWriter('out_trace.xlsx', engine='xlsxwriter')

for h in [8,9]:
    print('= = = = = = = = = = = = = = = =')
    print('Hour: ', h)
    with pm.Model() as model:    
        hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=5)
        hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=10)
        
        hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=5)
        hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=10)
        
        alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd)
        mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd)

        y_obs = data[data.Hour==h]['Connected'].values

        y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=y_obs)

        y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)

        trace = pm.sample(20, tune=1, chains=4, progressbar=False, nuts={"target_accept": 0.9})

        #trace24[h] = list(trace)
        ess = pm.diagnostics.effective_n(trace)

    print('- ESS: ', ess)
    obs = np.mean(data[data.Hour==h]['Connected'].values)
    print('- Observed: ', obs)
    #print(np.histogram(data[data.Hour==h]['Connected'].values, bins=bins16, density=True)[0])
    pred = np.mean(trace.get_values('y_pred'))
    print('- Predictive: ', np.mean(trace.get_values('y_pred')))
    #print(np.histogram(trace.get_values('y_pred'), bins=bins16, density=True)[0])
    print('Error: ', np.abs(pred-obs)/obs )

    out_trace = pd.DataFrame.from_dict(list(trace))
    name = 'out_hr' + str(int(h)) + '.csv'
    out_trace.to_csv(name)

    out_smry = pd.DataFrame(pm.summary(trace))
    name = 'out_hr' + str(int(h)) + '_smry.csv'
    out_smry.to_csv(name)

    out_yPred[h] = np.histogram(trace.get_values('y_pred'), bins=bins16)[0]
    out_yObs[h] = np.histogram(data[data.Hour==h]['Connected'].values, bins=bins16)[0]

out_yPred = pd.DataFrame(out_yPred)
out_yObs = pd.DataFrame(out_yObs)

# Export results
#out_trace = pd.DataFrame(out_trace)
#out_trace.to_csv('out_trace.csv')
#writer.save()
out_yObs.to_csv('out_yObs.csv')
out_yPred.to_csv('out_yPred.csv')
