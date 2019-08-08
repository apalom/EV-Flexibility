# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:53:36 2019

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborns as sns

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



#%%

import XlsxWriter

writer = pd.ExcelWriter('out_trace.xlsx', engine='xlsxwriter')

trace_data.to_excel(writer, sheet_name='trace_data01')

writer.save()