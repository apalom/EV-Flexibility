# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:01:18 2020

@author: Alex
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplot

cnctDays = pd.DataFrame(np.zeros((96,522)))

for j in range(522):
    print(j*96, j*96+96)
    cnctDays[j] = dfSLC_aggData.Connected[j*96:j*96+96].values



#%% Plot Hist
    
sns.set_style("whitegrid")
ax = sns.distplot(dfSLC_aggData.Connected, bins=np.arange(0,max(dfSLC_aggData.Connected)))

ax.set_title("EVs Connected")

ax.set(xlabel='Count', ylabel='Density')
plt.show()