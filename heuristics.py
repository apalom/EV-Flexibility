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
#cities = cities.drop('Unnamed: 0', axis=1)

cities = cities.loc[cities['2019 Population'] > 50000]

#%% Measure Distance 

def calcMinDist(loc,cities):
    
    from geopy import distance
    
    dist = np.zeros((len(cities),1))
    dist = cities['Lat-Lng'].apply(lambda x: distance.distance(loc,x).miles) 
    
    dist0 = np.min(dist)
    nearest = dist.idxmin() 
    nearestCity = cities.at[nearest,'City']
    population = cities.at[nearest,'2019 Population']
        
    return nearestCity, population, dist0;

#%% Create EVSE Heurstics DF 

def calcHeurDF(df, wknd):

    if wknd == 'Weekend':
        df = df.loc[df['DayofWk'] >= 5]
        print("--- Weekends ---")
    elif wknd == 'Weekday':
        df = df.loc[df['DayofWk'] < 5]
        print("--- Weekdays ---")
    elif wknd == 'All':
        print("--- All Days ---")
    
    EVSEs = list(set(df['EVSE ID']))
    
    dfEVSEs = pd.DataFrame(np.zeros((len(EVSEs),7)), index = EVSEs, 
                           columns=['Station','Port Type','Lat','Lng','Nearest','Population','Miles'])
    
    for e in EVSEs:
        station = df.loc[df['EVSE ID'] == e].iloc[0]['Station Name']
        port = df.loc[df['EVSE ID'] == e].iloc[0]['Port Type']
        lat = df.loc[df['EVSE ID'] == e].iloc[0].Latitude
        lng = df.loc[df['EVSE ID'] == e].iloc[0].Longitude
        loc = (lat,lng)
        
        # calculate nearest city to each EVSE station
        nearest, population, miles = calcMinDist(loc,cities)
        
        dfEVSEs.at[e] = [station,port,lat,lng,nearest,population,miles]        
        
    grouped = df.groupby('EVSE ID');
    
    for name, group in grouped:
        dfEVSEs.at[name,'SeshkWh'] = group['Energy (kWh)'].mean()
        dfEVSEs.at[name,'SeshPwr'] = group['AvgPwr'].mean()
        if len(group) == 1:
            dfEVSEs.at[name,'Age'] = 1;
            print('Single Charge Session')
        else:
            dfEVSEs.at[name,'Age'] = (group.iloc[len(group)-1]['Start Date'] - group.iloc[0]['Start Date']).days
        
        dfEVSEs.at[name,'daykWh'] = group['Energy (kWh)'].sum()/dfEVSEs.at[name,'Age']
        
        tChrg = group['Charging Time (hh:mm:ss)'].mean()        
        dfEVSEs.at[name,'SeshChrg'] = tChrg.seconds/3600
        
        tCnct = group['Total Duration (hh:mm:ss)'].mean()
        dfEVSEs.at[name,'SeshCnctd'] = tCnct.seconds/3600
        
        dfEVSEs.at[name,'Sparrow'] = dfEVSEs.at[name,'SeshChrg']/dfEVSEs.at[name,'SeshCnctd']
    
    return dfEVSEs;

#%% Build Heuristics Table
    
tHeur = timeit.default_timer()

#wknd = all, weekday or weekend    
dayFilter = 'All'
dfEVSEs = calcHeurDF(dfAll, dayFilter)

fileName = 'exports\\EVSE_Heuristics-' + dayFilter + '50k_pop.xlsx'
        
dfEVSEs.to_excel(fileName)

# timeit statement
elapsedMain = timeit.default_timer() - tHeur
print('Run time: {0:.4f} sec'.format(elapsedMain))
