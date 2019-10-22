#~/VENV3.6.3/bin/

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm

#%%
# When we want to understand the effect of more factors such as "day of week,"
# "time of day," etc. We can use GLM (generalized linear models) to better
# understand the effects of these factors.
print('Running ', str(datetime.now()))
# Import Data
data = pd.read_csv('hdc_wkdy20.csv',  index_col='Idx');

# Setup vars
out_trace = {}; out_yPred = {}; out_yObs = {}; 
smpls = 1000; tunes = 250; target = 0.9;

# Convert categorical variables to integer
le = preprocessing.LabelEncoder()
hr_idx = le.fit_transform(data.Hour)
hrs = le.classes_
n_hrs = len(hrs)

# Print Header
#print('hdc_wkdy20.csv | NB with Normal Prior')
print('hdc_wkdy20.csv | Poisson with U Prior')
print('Params: samples = ', smpls, ' | tune = ', tunes, ' | target = ', target, '\n')

if __name__ == "__main__":
#% Hierarchical Modeling
    with pm.Model() as model:
        hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=20)
        hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=20)
        #hyper_alpha_mu = pm.Normal('hyper_alpha_mu', mu=hr_std)
    
        hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=20)
        hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=20)
        #hyper_mu_mu = pm.Normal('hyper_mu_mu', mu=hr_mean)
    
        alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd, shape=n_hrs)
        mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd, shape=n_hrs)
    
        #alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd)
        #mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd)
    
        y_obs = data.Connected.values
    
        y_est = pm.Poisson('y_est', mu=mu[hr_idx], observed=y_obs)
        y_pred = pm.Poisson('y_pred', mu=mu[hr_idx], shape=data.Hour.shape)
    
        #y_est = pm.NegativeBinomial('y_est', mu=mu[hr_idx], alpha=alpha[hr_idx], observed=y_obs)
        #y_pred = pm.NegativeBinomial('y_pred', mu=mu[hr_idx], alpha=alpha[hr_idx], shape=data.Hour.shape)
    
        trace = pm.sample(smpls, tune=tunes, cores=1, chains=4, progressbar=True, nuts={"target_accept": target})
    
        # Export traceplotstr(int(h)) +
        #trarr = pm.traceplot(trace[tunes:])
        #fig = plt.gcf()
        #fig.savefig("out_hr" + str(int(h)) + "_tracePlt" + ".png")

pm.save_trace(trace, 'results/NBsmpls' + str(int(smpls)) + '.trace')

ess = pm.diagnostics.effective_n(trace)

print('- ESS: ', ess)

out_trace = pd.DataFrame.from_dict(list(trace))
out_trace.to_csv('results/out_trace.csv')

out_smry = pd.DataFrame(pm.summary(trace))
out_smry.to_csv('results/out_smry.csv')
