# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:53:36 2019

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import nbinom, gamma, poisson

# Read in trace data, hourly data seperated by sheet

path = 'results/1188820_trace_smry_xlsx_500smpl/out_trace.xlsx'

hours = np.arange(0,24)

trace24_dict = {}
trace24_smry_dict = {}

for h in hours:
    sheet = 'hour'+str(h)
    trace24_dict[h] = pd.read_excel(path, index_col=[0], sheetname=sheet)
    
    sheet = 'hour'+str(h)+'_smry'
    trace24_smry_dict[h] = pd.read_excel(path, index_col=[0], sheetname=sheet)
    
#%% Plot Hourly Value Distributions
# https://docs.pymc.io/notebooks/GLM-negative-binomial-regression.html

def get_nb_vals(mu, alpha, size):
    """Generate negative binomially distributed samples by drawing a sample from a gamma 
    distribution with mean `mu` and shape parameter `alpha', then drawing from a Poisson
    distribution whose rate parameter is given by the sampled gamma variable."""    

    g = stats.gamma.rvs(alpha, scale=mu / alpha, size=size)
    return stats.poisson.rvs(g)

mu = trace_smry['mean']['mu']
alpha = trace_smry['mean']['alpha']

plt.hist(get_nb_vals(mu, alpha, 1000), bins=np.arange(0,16), density=True, edgecolor='white', linewidth=1.2, label='Connected')
#plt.set(xticks=np.arange(0,26,2), xlim=[-1, 25])
plt.title('NegBino Trace')

#%% Effective Sample Size 

dctData = {}
dctSmry = {}

allMu = pd.DataFrame(np.zeros((0,0)))
allYpred = pd.DataFrame(np.zeros((0,0)))

for h in hours:
    
    # Read in ALL TRACE data
    name = 'results/1190449_15000smpl_1000tune/out_hr'+str(h)+'.csv'
    dctData[h] = pd.read_csv(name, index_col=[0])
    allMu = pd.concat([allMu, dctData[h].mu], ignore_index=True)    
    allYpred = pd.concat([allYpred, dctData[h].y_pred], ignore_index=True)    
    
    # Read in summary trace data
    name = 'results/1190449_15000smpl_1000tune/out_hr'+str(h)+'_smry.csv'
    dctSmry[h] = pd.read_csv(name, index_col=[0])

ess = np.var(allYpred.values)/np.var(allMu.values)
print('Effective Sample Size: ', ess)

#%%

fig = plt.figure(figsize=(10,6))

fig.add_subplot(211)
binsCnt = np.arange(0,np.max(allMu.values))
plt.hist(allMu.values, bins=binsCnt, edgecolor='white')
plt.title('All Mean Value Histogram')

fig.add_subplot(212)
bins = np.arange(0,np.max(allYpred.values))
plt.hist(allYpred.values, bins=binsCnt, color='red', edgecolor='white')
plt.title('All y_pred Value Histogram')

plt.tight_layout()


#%% 

import XlsxWriter

writer = pd.ExcelWriter('out_trace.xlsx', engine='xlsxwriter')

trace_data.to_excel(writer, sheet_name='trace_data01')

writer.save()