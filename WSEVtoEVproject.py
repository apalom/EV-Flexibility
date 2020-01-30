# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:56:27 2020

@author: Alex Palomino
"""

import pandas as pd

#dataWSEV = pd.read_excel("evolution\data_WSEV2019.xlsx", index_col=[0]);
dataINL = pd.read_excel("evolution\data_EV2011.xlsx")

#dataINL = dataINL.loc[dataINL['GroupBy'] == 'Away']
dataINL_cht = dataINL.loc[dataINL['ChtName'] == 'DistributionACEnergyConsumedPerChargeEvent']
dataINL_cht = dataINL_cht.loc[dataINL['CatName'] == 'WD']

dataWSEV = dataWSEV.loc[dataWSEV['Port Type'] == 'Level 2']

#%% Session Energy Histogram

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", font='Times New Roman')
plt.figure(figsize=(12,8))

bns = np.arange(0,27,3); bns1 = np.arange(0,36,3)
sns.distplot(dataWSEV['Energy (kWh)'], bins=bns1, kde=False, rug=False,norm_hist=True, label='WSEV');
sns.lineplot(bns, dataINL_cht.ReportValue, color='r', label='EVproj')

plt.title('L2 Session Energy Density')
plt.ylabel('kWh')
plt.xticks(bns1)
plt.legend()

#%% Energy Demand Per Time of Day 