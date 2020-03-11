# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:18:43 2020

@author: Alex Palomino
"""

import pandas as pd
import numpy as np
from itertools import combinations 

chgrDistances = pd.read_excel('data/charger_distances.xlsx');

#%% Basic Driver Metrics

#dfTemp = dict_seshDrvr['3820891']
#plt.scatter(dfTemp['Start Date'], dfTemp['Energy (kWh)'])# s=dfTemp['Energy (kWh)'])

df_metrics = pd.DataFrame(np.empty((len(drvr_include),6)), 
                          index=drvr_include, columns=['AvgEnergy','AvgDur','AvgChrg','Ratio','AvgStart','NumChgrs'])

for d in drvr_include:
    df_metrics.AvgEnergy.at[d] = dict_seshDrvr[d]['Energy (kWh)'].mean()
    df_metrics.AvgDur.at[d] = dict_seshDrvr[d]['Duration (h)'].mean()
    df_metrics.AvgChrg.at[d] = dict_seshDrvr[d]['Charging (h)'].mean()
    df_metrics.Ratio.at[d] = df_metrics.AvgChrg.at[d]/df_metrics.AvgDur.at[d]
    df_metrics.AvgStart.at[d] = dict_seshDrvr[d]['StartHr'].mean()
    df_metrics.NumChgrs.at[d] = len(list(set(dict_seshDrvr[d]['EVSE ID'])))

print('--- Summary ---\n',df_metrics.describe())

#%% Calculate Maximum Distance between Chargers Used by a Single Driver

dict_maxDist = {};
for d in drvr_include:
    
    EVSE_pairs = list(combinations(list(set(dict_seshDrvr[d]['EVSE ID'])),2))
    
    dist = {};
    for p in EVSE_pairs:
        try:
            if len(chgrDistances.loc[chgrDistances['EVSE ID'] == str(p)]) == 1:
                dist[p] = float(chgrDistances.loc[chgrDistances['EVSE ID'] == str(p)].Dist.values[0].split()[0])
            else:
                dist[p] = float(chgrDistances.loc[chgrDistances['EVSE ID'] == str(p[::-1])].Dist.values[0].split()[0])
        
            if len(dist) > 0:
                dict_maxDist[d] = max(dist.values())     
            else: 
                dict_maxDist[d] = 'EVSE pair not found.'
            
        except ValueError:
            dict_maxDist[d] = 'EVSE pair not found.'
    
    print('Driver :', d, ' | ', dict_maxDist[d])

    

#%%
