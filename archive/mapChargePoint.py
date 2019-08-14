# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:00:32 2019

@author: Alex
"""

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='apalom', api_key='xSoBctgLdCd3w4Qn75F2')

mapbox_access_token = 'pk.eyJ1IjoiYXBhbG9tIiwiYSI6ImNpcW50YzBjazAxYm5mcW5odnN0dXNkOGUifQ.l8TWN6aZcxxSp-rcZ717qQ'

data = [
    go.Scattermapbox(
        lat=['45.5017'],
        lon=['-73.5673'],
        mode='markers',
        marker=dict(
            size=14
        ),
        text=['Montreal'],
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=45,
            lon=-73
        ),
        pitch=0,
        zoom=5
    ),
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='Montreal Mapbox')

#%%

import plotly.plotly as py
import pandas as pd

df = pd.read_csv(r'C:\Users\Alex\Box Sync\Alex and Masood\WestSmartEV\Data from Partners\ChargePoint Data\publicEVSEs.csv')

for col in df.columns:
    df[col] = df[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df['text'] = df['Station Name'] + '<br>' +\
    'Port Type' + df['Port Type'] + '<br>' +\
    'Daily Energy '+ df['Avg Energy']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = list(zip(df['Latitude'],df['Longitude'])),
        z = df['Energy (kWh)'].astype(float),
        locationmode = 'USA-states',
        text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Millions USD")
        ) ]

layout = dict(
        title = '2011 US Agriculture Exports by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.plot( fig, filename='d3-cloropleth-map' )
