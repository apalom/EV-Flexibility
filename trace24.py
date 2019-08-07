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
data = pd.read_csv('hr_day_cnctd.csv',  index_col='Index');

#%% Houry NegativeBinomial Modelings
# For each hour j and each EV connected i, we represent the model
indiv_traces = {};

# Convert categorical variables to integer
hours = list(data.Hour)
n_hours = len(hours)
x_lim = 16

out_yPred = pd.DataFrame(np.zeros((1,n_hours)), columns=list(hours))
out_yObs = pd.DataFrame(np.zeros((1,n_hours)), columns=list(hours))

for h in hours:
    print('Hour: ', h)
    with pm.Model() as model:
        alpha = pm.Uniform('alpha', lower=0, upper=10)
        mu = pm.Uniform('mu', lower=0, upper=10)

        y_obs = data[data.Hour==h]['Connected'].values
        y_est = pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=y_obs)

        y_pred = pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha)

        trace = pm.sample(100, progressbar=True)

        #indiv_traces[h] = trace

    print('--- Observed ---')
    print(np.histogram(data[data.Hour==h]['Connected'].values, bins=x_lim))
    print('--- Predictive ---')
    print(np.histogram(trace.get_values('y_pred'), bins=x_lim))

    [out_yPred.loc[:,h], _] = np.histogram(trace.get_values('y_pred'), bins=x_lim)
    [out_yObs.loc[:,h], _] = np.histogram(data[data.Hour==h]['Connected'].values, bins=x_lim)

# Export results
out_yPred.to_csv('out_yPred.csv')
out_yObs.to_csv('out_yObs.csv')
