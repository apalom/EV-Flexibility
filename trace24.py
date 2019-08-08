#~/VENV3.6.3/bin/

# import sys
# sys.path.append('/uufs/chpc.utah.edu/sys/installdir/python/3.6.3/lib/python3.6/site-packages/')
# import os
# sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
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
data = pd.read_csv('hr_day_cnctd_wkdy.csv',  index_col='Idx');

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

writer = pd.ExcelWriter('out_trace.xlsx', engine='xlsxwriter')

for h in hours:
    print('Hour: ', h)
    with pm.Model() as model:
        alpha = pm.Uniform('alpha', lower=0, upper=10)
        mu = pm.Uniform('mu', lower=0, upper=10)

        y_obs = data[data.Hour==h]['Connected'].values

        y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=y_obs)

        y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)

        trace = pm.sample(200000, tune=10000, progressbar=False)

        #trace24[h] = list(trace)

    print('--- Observed ---')
    print(np.histogram(data[data.Hour==h]['Connected'].values, bins=bins16, density=True)[0])
    print('--- Predictive ---')
    print(np.histogram(trace.get_values('y_pred'), bins=bins16, density=True)[0])

    out_trace = pd.DataFrame.from_dict(list(trace))
    out_smry = pd.DataFrame(pm.summary(trace))
    name = 'hour' + str(int(h))
    out_trace.to_excel(writer, sheet_name=name)
    name = 'hour' + str(int(h)) + '_smry'
    out_smry.to_excel(writer, sheet_name=name)


    out_yPred[h] = np.histogram(trace.get_values('y_pred'), bins=bins16)[0]
    out_yObs[h] = np.histogram(data[data.Hour==h]['Connected'].values, bins=bins16)[0]

out_yPred = pd.DataFrame(out_yPred)
out_yObs = pd.DataFrame(out_yObs)

# Export results
#out_trace = pd.DataFrame(out_trace)
#out_trace.to_csv('out_trace.csv')
writer.save()
out_yObs.to_csv('out_yObs.csv')
out_yPred.to_csv('out_yPred.csv')
