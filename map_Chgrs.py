# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:45 2019

@author: Alex
"""

from geopy import geocoders  
import pandas as pd
import numpy as np
import googlemaps
import time
from itertools import combinations 

#%% Import Data
def loadData():
    # Import Data
    filePath = 'data/Session-Details-Summary-20200113.csv';
    return pd.read_csv(filePath);

#%%
df_Chgrs = loadData()[['EVSE ID','Port Number','Port Type','Station Name','Latitude','Longitude']].drop_duplicates() 
df_Chgrs = df_Chgrs.set_index(df_Chgrs['EVSE ID'],drop=True).drop(['EVSE ID'], axis=1)
df_Chgrs = df_Chgrs.loc[~df_Chgrs.index.duplicated(keep='first')]

# Create distance matrix
df_ChgrDist = pd.DataFrame((np.empty((len(df_Chgrs),len(df_Chgrs)))), columns=df_Chgrs.index, index=df_Chgrs.index)

#%% Calculate Distances

# Call Google API
gn = geocoders.GeoNames(username='apalom')
googleKey = 'AIzaSyCKL5hpSAGfGa0ZplzLKyVbA3oN3-FKjTI'
gmaps = googlemaps.Client(key=googleKey)

latlons = list(combinations(df_ChgrDist.index,2))

dist_Dict = {}
for pair in latlons:
    loc1 = (df_Chgrs.Latitude.at[pair[0]], df_Chgrs.Longitude.at[pair[0]])
    loc2 = (df_Chgrs.Latitude.at[pair[1]], df_Chgrs.Longitude.at[pair[1]])
    dist_Dict[pair] = gmaps.distance_matrix(loc1,loc2)['rows'][0]['elements'][0]['distance']['text']
    print(pair, dist_Dict[pair])

#%% Geocoding Addresses
for index, row in df_ChgrDist.iterrows():  # assumes mydata is a pandas df
    time.sleep(0.5)
    geocode_result = gmaps.geocode(row.City)
    lat = geocode_result[0]['geometry']['location']['lat']
    lng = geocode_result[0]['geometry']['location']['lng']
    print(row.City, lat, lng)
    #geocoded.append(latlon)  # geocode function returns a geocoded object
    dfCities['Lat'].at[index] = lat;
    dfCities['Lng'].at[index] = lng;
    