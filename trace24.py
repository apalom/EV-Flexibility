#~/VENV3.6.3/bin/

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import pymc3 as pm
#%%
# When we want to understand the effect of more factors such as "day of week,"
# "time of day," etc. We can use GLM (generalized linear models) to better
# understand the effects of these factors.
print('Running ', str(datetime.now()))
# Import Data
data = pd.read_csv('hdc_wkdy20.csv',  index_col='Idx');

# Setup vars
hours = np.arange(0,24)
#hours = [0,4,8,12,16,20]
out_trace = {}; out_yPred = {}; out_yObs = {}; trace24 = {};
smpls = 1000; tunes = 50; target = 0.9;

# Select Parameters from data
#params = pd.DataFrame(columns=['mu', 'stddev'])
#for h in hours:
#    dfTemp = data.loc[data.Hour == h]
#    params.loc[h] = [np.mean(dfTemp.Connected), np.std(dfTemp.Connected)]

# Print Header
#print('hdc_wkdy20.csv | NB with Normal Prior')
print('hdc_wkdy20.csv | Poisson with Normal Prior')
print('Params: samples = ', smpls, ' | tune = ', tunes, ' | target = ', target, '\n')

#%% Houry Modeling
for h in hours:
    print('= = = = = = = = = = = = = = = =')
    print('Hour: ', h)

    dfTemp = data.loc[data.Hour == h]
    hr_mean = np.mean(dfTemp.Connected)
    hr_std = np.std(dfTemp.Connected)

    with pm.Model() as model:
        hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=20)
        hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=20)
        #hyper_alpha_mu = pm.Normal('hyper_alpha_mu', mu=hr_std)

        hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=20)
        hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=20)
        #hyper_mu_mu = pm.Normal('hyper_mu_mu', mu=hr_mean)

        alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd)
        mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd)

        y_obs = data[data.Hour==h]['Connected'].values

        y_est = pm.Poisson('y_est', mu=mu, observed=y_obs)
        y_pred = pm.Poisson('y_pred', mu=mu)

        #y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=y_obs)
        #y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)

        trace = pm.sample(smpls, tune=tunes, chains=4, progressbar=False, nuts={"target_accept": target})

        # Export traceplotstr(int(h)) +
        #trarr = pm.traceplot(trace[tunes:])
        #fig = plt.gcf()
        #fig.savefig("out_hr" + str(int(h)) + "_tracePlt" + ".png")

        #pm.save_trace(trace, 'out_hr' + str(int(h)) + '.trace')

        #trace24[h] = list(trace)
        #trace24[h] = pm.save_trace(trace)
        ess = pm.diagnostics.effective_n(trace)

    print('- ESS: ', ess)
    obs = np.mean(data[data.Hour==h]['Connected'].values)
    print('- Observed: ', obs)
    pred = np.mean(trace.get_values('y_pred'))
    print('- Predictive: ', pred)
    print('Error: ', np.abs(pred-obs)/obs )

    out_trace = pd.DataFrame.from_dict(list(trace))
    name = 'out_hr' + str(int(h)) + '_trace.csv'
    out_trace.to_csv(name)

    out_smry = pd.DataFrame(pm.summary(trace))
    name = 'out_hr' + str(int(h)) + '_smry.csv'
    out_smry.to_csv(name)

    out_yPred[h] = np.histogram(trace.get_values('y_pred'), bins=np.arange(0,16))[0]
    out_yObs[h] = np.histogram(data[data.Hour==h]['Connected'].values, bins=np.arange(0,16))[0]

out_yPred = pd.DataFrame(out_yPred)
out_yObs = pd.DataFrame(out_yObs)

# Export results
out_yObs.to_csv('out_yObs.csv')
out_yPred.to_csv('out_yPred.csv')
