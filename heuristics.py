# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:08:40 2019

@author: Alex Palomino
"""

# Import Libraries
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import timeit
import time
import datetime


#%% Import Data

# Raw Data
filePath = 'data/Session-Details-Summary-20190404.csv';

# Import Data
data = pd.read_csv(filePath);

#dataHead = data.head(100);
#dataTypes = data.dtypes;

allColumns = list(data);

#%%

from stochProcess import filterPrep

dfAll, daysTot = filterPrep(data, "PACKSIZE", False)

#%%
dfHeur = pd.DataFrame(dfAll, columns=['Station Name', 'EVSE ID', 'Energy (kWh)', 
                                      'Charging Time (hh:mm:ss)', 'Total Duration (hh:mm:ss)', 
                                      'Port Type', 'Latitude', 'Longitude'])

#%% Import City Location Data

# Raw Data
filePath = 'data/UT-citiesGPS.csv';

# Import Data
cities = pd.read_csv(filePath);
cities = cities.drop(columns='Unnamed: 0')

#%% Create EVSE ID 

EVSEs = list(set(dfAll['EVSE ID']))

dfEVSEs = pd.DataFrame(np.zeros((len(EVSEs),2)),index = EVSEs, columns=['Lat','Lng'])

for e in EVSEs:
    lat = dfAll.loc[dfAll['EVSE ID'] == e].iloc[0].Latitude
    lng = dfAll.loc[dfAll['EVSE ID'] == e].iloc[0].Longitude
    dfEVSEs.at[e] = [lat,lng]


#%% Measure Distance 
