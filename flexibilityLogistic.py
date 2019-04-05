# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:53:03 2019

@author: Alex Palomino
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import mstats
import matplotlib.pyplot as plt
import timeit
import time
import datetime


import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression

data = pd.read_csv('data/Lifetime-Session-Details.csv')


#%% Data Columns for ChargePoint 'data/Lifetime-Session-Details.csv';

colNames = ['EVSE ID', 'Port Number', 'Station Name', 'Plug In Event Id', 'Start Date', 'End Date', 
            'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)', 'Energy (kWh)',
            'Ended By', 'Port Type', 'Latitude', 'Longitude', 'User ID', 'Driver Postal Code'];
            
data = pd.DataFrame(data, index=np.arange(len(data)), columns=colNames)

data['Start Date'] = pd.to_datetime(data['Start Date']);
data['End Date'] = pd.to_datetime(data['End Date']);
data['Total Duration (hh:mm:ss)'] = pd.to_timedelta(data['Total Duration (hh:mm:ss)']);
data['Charging Time (hh:mm:ss)'] = pd.to_timedelta(data['Charging Time (hh:mm:ss)']);

dataHead = data.head(100);

#%% All EVSEs

dfAll = data.loc[data['Energy (kWh)'].notna()]
dfAll['Duration (h)'] = dfAll['Total Duration (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
dfAll['Duration (h)'] = dfAll['Duration (h)'].apply(lambda x: round(x * 2) / 4) 
dfAll['Charging (h)'] = dfAll['Charging Time (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
dfAll['Charging (h)'] = dfAll['Charging (h)'].apply(lambda x: round(x * 2) / 4) 

dfAll['DayofYr'] = dfAll['Start Date'].apply(lambda x: x.dayofyear) 
dfAll['DayofWk'] = dfAll['Start Date'].apply(lambda x: x.weekday()) 
dfAll['StartHr'] = dfAll['Start Date'].apply(lambda x: x.hour + x.minute/60) 
dfAll['StartHr'] = dfAll['StartHr'].apply(lambda x: round(x * 4) / 4) 
dfAll['EndHr'] = dfAll['End Date'].apply(lambda x: x.hour + x.minute/60) 
dfAll = dfAll.loc[dfAll['EndHr'].notna()]
dfAll['EndHr'] = dfAll['EndHr'].apply(lambda x: round(x * 4) / 4) 
dfAll['AvgPwr'] = dfAll['Energy (kWh)']/dfAll['Duration (h)']

dfAll = dfAll.loc[dfAll['Duration (h)'] > 0]
dfAll = dfAll.sort_values(by=['Start Date']);
dfAll = dfAll.reset_index(drop=True);

dfHead = dfAll.head(100);

#%% Calc Connected Time

i=0;
bW = 0.25;
binEdges = np.arange(0,24.25,bW);
#allEVSEs = list(set(dfAll['EVSE ID']));
allEVSEs = [121987]

for EVSE in allEVSEs:
    
    dfTemp = dfAll.loc[dfAll['EVSE ID'] == EVSE]  
    dfTemp = dfTemp.sort_values(by=['Start Date']);   
    
    
    
    dfTemp1 = dfTemp.loc[dfTemp['Port Number'] == '1']            
#        
#    dfTemp2 = dfTemp.loc[dfTemp['Port Number'] == '2']
#    dfTemp2 = dfTemp2.sort_values(by=['StartHr']);  
    
    daysAlive = (dfTemp1['Start Date'].iloc[len(dfTemp1)-1] - dfTemp1['Start Date'].iloc[0]).days
    
# --- Connected ---
    p_connected = np.zeros((len(binEdges)-1,len(dfTemp1)));
    #conn = np.arange(0);
    for s in range(len(dfTemp1)-1):  
        conn = np.zeros((len(binEdges)-1,1));
        st = dfTemp.StartHr.iloc[s]
        en = st + dfTemp['Duration (h)'].iloc[s]
        stIdx = int(st/0.25);
        enIdx = int(en/0.25);        
        if en >= 23.75:
            print('[--- MIDNIGHT ---]')
            en1 = 23.75
            enIdx1 = int(en1/0.25)
            en2 = en - 23.75
            enIdx2 = int(en2/0.25)
            conn[stIdx:enIdx1] = 1; 
            conn[0:enIdx2] = 1;
        else:
            conn[stIdx:enIdx] = 1; 

        p_connected[:,i] = conn[:,0]
    
        i += 1;

    noChargeDays = np.zeros((len(binEdges)-1,daysAlive));
    
    