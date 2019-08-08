# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:53:36 2019

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborns as sns
import scipy.stats as stats
from scipy.stats import nbinom, gamma, poisson

# Read in trace data, hourly data seperated by sheet
hours = np.linspace(0,23,24)

trace24_dict = {}
trace24_smry_dict = {}

for h in hours:
    sheet = 'hour'+str(int(h))+'.0'
    trace24_dict[h] = pd.read_excel('results/1188783_trace_xlsx_500smpl_25tune/out_trace.xlsx',  index_col=[0], sheetname=sheet)
    
    #sheet = 'hour'+str(int(h))+'.0_smry'
    #trace24_smry_dict[h] = pd.read_excel('results/1188783_trace_xlsx_500smpl_25tune/out_trace.xlsx',  index_col=[0], sheetname=sheet)
    
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

#%%

import XlsxWriter

writer = pd.ExcelWriter('out_trace.xlsx', engine='xlsxwriter')

trace_data.to_excel(writer, sheet_name='trace_data01')

writer.save()