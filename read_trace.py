# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:53:36 2019

@author: Alex
"""

import pandas as pd
import XlsxWriter

trace_data = pd.read_csv('results/out_trace.csv',  index_col='smplNo');

writer = pd.ExcelWriter('out_trace.xlsx', engine='xlsxwriter')
trace_data.to_excel(writer, sheet_name='trace_data01')

