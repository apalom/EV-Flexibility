# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:45 2019

@author: Alex
"""

from geopy import geocoders  
import pandas as pd
import googlemaps
import time


gn = geocoders.GeoNames(username='apalom')
googleKey = 'AIzaSyBHTjbBLRMvxLEpgLQDan33Zrk3a04fa54'


# Import Data
filePath = 'data/UT-cities.csv';
dfCities = pd.read_csv(filePath);

dfCities = pd.DataFrame(dfCities, columns=['City', '2019 Population', 'Lat', 'Lng'])

gmaps = googlemaps.Client(key=googleKey)

# Geocoding an address

for index, row in dfCities.iterrows():  # assumes mydata is a pandas df
    time.sleep(0.5)
    geocode_result = gmaps.geocode(row.City)
    lat = geocode_result[0]['geometry']['location']['lat']
    lng = geocode_result[0]['geometry']['location']['lng']
    print(row.City, lat, lng)
    #geocoded.append(latlon)  # geocode function returns a geocoded object
    dfCities['Lat'].at[index] = lat;
    dfCities['Lng'].at[index] = lng;
