# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:24:18 2019

@author: Alex

Prepare test/training data
"""


# Import Libraries
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import datetime
from sklearn.model_selection import train_test_split, KFold

#%% Import Data

def loadData():
    # Import Data
    filePath = 'data/Session-Details-Summary-20200113.csv';
    return pd.read_csv(filePath);

#% Dataframe Preparation

def filterPrep(df, string, fltr, time):

    colNames = ['EVSE ID', 'Port Number', 'Port Type', 'Station Name', 'Plug In Event Id', 'City', 'Latitude', 'Longitude',
                'User ID', 'Driver Postal Code', 'Start Date', 'End Date', 'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)',
                'Energy (kWh)', 'Ended By', 'Start SOC', 'End SOC'];

    df = pd.DataFrame(df, index=np.arange(len(df)), columns=colNames)
    
    #filter for dfcf
    #df = df.loc[df['Port Type'] == 'DC Fast']

    df['Start Date'] = pd.to_datetime(df['Start Date']);
    df['End Date'] = pd.to_datetime(df['End Date']);
    df['Total Duration (hh:mm:ss)'] = pd.to_timedelta(df['Total Duration (hh:mm:ss)']);
    df['Charging Time (hh:mm:ss)'] = pd.to_timedelta(df['Charging Time (hh:mm:ss)']);

    #filter by City
    if fltr:
        df = df[df['City'].str.contains(string)]
        print("Filter for: ", string)
    else:
        print("No Filter")

    #clean data
    df = df.loc[df['Energy (kWh)'] > 0]
    df = df.loc[~pd.isnull(df['End Date'])]
    yr = 2019
    df = df.loc[(df['Start Date'] > datetime.date(yr,1,1)) & (df['Start Date'] < datetime.date(yr+1,1,1))]

    #update data types
    df['Duration (h)'] = df['Total Duration (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
    #df['Duration (h)'] = df['Duration (h)'].apply(lambda x: round(x * 4) / 4)
    df['Charging (h)'] = df['Charging Time (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
    #df['Charging (h)'] = df['Charging (h)'].apply(lambda x: round(x * 4) / 4)
    df['NoCharge (h)'] = df['Duration (h)'] - df['Charging (h)']
    df = df.loc[df['Duration (h)'] > 0]

    # Day of year 0 = Jan1 and day of year 365 = Dec31
    df['DayofYr'] = df['Start Date'].apply(lambda x: x.dayofyear)
    # Monday is 0 and Sunday is 6
    df['DayofWk'] = df['Start Date'].apply(lambda x: x.weekday())
    # Filter for weekdays
    df = df.loc[df['DayofWk'] <= 4]
    #df['isWeekday'] = df['DayofWk'].apply(lambda x: 1 if x <=4 else 0)
    #df = df.loc[df['isWeekday'] == 1]
    df['Year'] = df['Start Date'].apply(lambda x: x.year)
    df['StartHr'] = df['Start Date'].apply(lambda x: x.hour + x.minute/60)
    df['EndHr'] = df['End Date'].apply(lambda x: x.hour + x.minute/60)
    if time == '1hr':
        df['StartHr'] = df['StartHr'].apply(lambda x: np.round(x));
        df['EndHr'] = df['EndHr'].apply(lambda x: np.round(x));
    elif time == '15min':
        df['StartHr'] = df['StartHr'].apply(lambda x: round(x * 4) / 4)
        df['EndHr'] = df['EndHr'].apply(lambda x: round(x * 4) / 4)
    elif time == '5min':
        df['StartHr'] = df['StartHr'].apply(lambda x: round(x * 12) / 12)
        df['EndHr'] = df['EndHr'].apply(lambda x: round(x * 12) / 12)
    df['AvgPwr'] = df['Energy (kWh)']/df['Duration (h)']
    df['Date'] = df['Start Date'].apply(lambda x: str(x.year) + '-' + str(x.month) + '-' + str(x.day))

    #convert percent to float
    def p2f(s):
        if isinstance(s, str):
            x = s.strip('%')
            x = float(x)/100
            return x
        else:
            return s

    df['Start SOC'] =  df['Start SOC'].apply(lambda x: p2f(x))
    df['End SOC'] =  df['End SOC'].apply(lambda x: p2f(x))

    # Sort Dataframe
    df.sort_values(['Start Date'], inplace=True);
    df = df.reset_index(drop=True);

    # Assign Day Count
    df['dayCount'] = 0;

    days = list(df['Start Date'].apply(lambda x: str(x.year) + '-' + str(x.month) + '-' + str(x.day)))
    daysSet = sorted(set(days), key=days.index)

    c = 0;
    for d in daysSet:
        dateTest = [df['Date'] == d]
        trueIdx = list(dateTest[0][dateTest[0]].index)
        df.at[trueIdx,'dayCount'] = c
        c += 1;  

    return df;

# Salt Lake City Sessions
dfSLC_sesh = filterPrep(loadData(), "Salt Lake City", True, '15min')

#dfSLC_sesh.to_excel("evolution/data_WSEV2019.xlsx")
#%% Single Port
dfSLC_sesh1chgr = dfSLC_sesh.loc[dfSLC_sesh['EVSE ID'] == 167437]
dfSLC_sesh1port = dfSLC_sesh1chgr.loc[dfSLC_sesh1chgr['Port Number'] == str(1)]

#%% Single Driver
drivers = list(set(dfSLC_sesh['User ID']))
dict_drvr = {}

for i in range(len(drivers)):    
    dict_drvr[drivers[i]] = len(dfSLC_sesh.loc[dfSLC_sesh['User ID'] == drivers[i]])
    
drvr_include = [k for k,v in dict_drvr.items() if v >= 100]

dict_seshDrvr = {};
for d in drvr_include:
    print(d)
    dict_seshDrvr[d] = dfSLC_sesh.loc[dfSLC_sesh['User ID'] == str(d)]

#%% Basic Driver Metrics

plt.scatter(dfTemp['Start Date'], dfTemp['Energy (kWh)'])
    

#%% Calculate per time period values

def intervalData(df, weekday, ppD):

    dctDays = {};

    daysIn = list(set(df.dayCount))
    daysIn.sort()

    dfArrivals = pd.DataFrame(np.zeros((ppD,len(set(df.dayCount)))),
                          index=np.arange(0,ppD), columns=daysIn)
    
    dfDepartures = pd.DataFrame(np.zeros((ppD,len(set(df.dayCount)))),
                      index=np.arange(0,ppD), columns=daysIn)

    dfEnergy = pd.DataFrame(np.zeros((ppD,len(set(df.dayCount)))),
                          index=np.arange(0,ppD), columns=daysIn)
    
    dfEnergyTot = pd.DataFrame(np.zeros((ppD,len(set(df.dayCount)))),
                      index=np.arange(0,ppD), columns=daysIn)

    dfDuration = pd.DataFrame(np.zeros((ppD,len(set(df.dayCount)))),
                          index=np.arange(0,ppD), columns=daysIn)

    dfCharging = pd.DataFrame(np.zeros((ppD,len(set(df.dayCount)))),
                          index=np.arange(0,ppD), columns=daysIn)

    for d in df.dayCount:
        print('Day: ', d)
        dfDays = df[df.dayCount == d]
        
        # Count Arrivals/Departures
        arvl = dfDays.StartHr.value_counts();
        dept = dfDays.EndHr.value_counts();
        if ppD == 96:
            arvl.index = (arvl.index.values*4).astype(int);
            dept.index = (dept.index.values*4).astype(int);
        if ppD == 288:
            arvl.index = (arvl.index.values*12).astype(int);
            dept.index = (dept.index.values*12).astype(int);
        arvl = arvl.sort_index()
        dept = dept.sort_index()
       
        # Setup data
        tot_energy = dfDays['Energy (kWh)'].groupby(dfDays.StartHr).sum()
        sesh_energy = dfDays['Energy (kWh)'].groupby(dfDays.StartHr).mean()
        duration = dfDays['Duration (h)'].groupby(dfDays.StartHr).mean()
        charging = dfDays['Charging (h)'].groupby(dfDays.StartHr).mean()
        
        if ppD == 96: # 15 min data
            tot_energy.index = (tot_energy.index.values*4).astype(int);
            sesh_energy.index = (sesh_energy.index.values*4).astype(int);
            duration.index = (duration.index.values*4).astype(int);
            charging.index = (charging.index.values*4).astype(int);
        elif ppD == 288: # 5 min data
            tot_energy.index = (tot_energy.index.values*12).astype(int);
            sesh_energy.index = (sesh_energy.index.values*12).astype(int);
            duration.index = (duration.index.values*12).astype(int);
            charging.index = (charging.index.values*12).astype(int);
                
        dfArrivals.loc[:,d] = arvl
        dfArrivals.loc[:,d] = np.nan_to_num(dfArrivals.loc[:,d])
        
        dfDepartures.loc[:,d] = dept
        dfDepartures.loc[:,d] = np.nan_to_num(dfDepartures.loc[:,d])
                
        dfEnergy.loc[:,d] = sesh_energy
        dfEnergy.loc[:,d] = np.nan_to_num(dfEnergy.loc[:,d])
        
        dfEnergyTot.loc[:,d] = tot_energy
        dfEnergyTot.loc[:,d] = np.nan_to_num(dfEnergyTot.loc[:,d])

        dfDuration.loc[:,d] = duration
        dfDuration.loc[:,d] = np.nan_to_num(dfDuration.loc[:,d])

        dfCharging.loc[:,d] = charging
        dfCharging.loc[:,d] = np.nan_to_num(dfCharging.loc[:,d])

    dctDays['Arrivals'] = dfArrivals;
    dctDays['Departures'] = dfDepartures;
    dctDays['EnergyAvg'] = dfEnergy;
    dctDays['EnergyTot'] = dfEnergyTot;
    dctDays['Duration'] = dfDuration;
    dctDays['Charging'] = dfCharging;

    return dctDays

dfSLC_dayData = intervalData(dfSLC_sesh, True, 96)

#%%

dfSLC_dayData['Arrivals'].to_excel("evolution/data_WSEV2019_Arrivals.xlsx")
dfSLC_dayData['Departures'].to_excel("evolution/data_WSEV2019_Departures.xlsx")
dfSLC_dayData['EnergyAvg'].to_excel("evolution/data_WSEV2019_EnergyAvg.xlsx")
dfSLC_dayData['EnergyTot'].to_excel("evolution/data_WSEV2019_EnergyTot.xlsx")
dfSLC_dayData['Duration'].to_excel("evolution/data_WSEV2019_Duration.xlsx")
dfSLC_dayData['Charging'].to_excel("evolution/data_WSEV2019_Charging.xlsx")


#%% Save
per = '1hr'

#dfSLC_dayData.to_excel("data/dfSLC_dayData_2018-2019.xlsx")
dfSLC_dayData['Arrivals'].to_excel("data/"+per+"/dfArrivals_dayData_2018-2019.xlsx")
dfSLC_dayData['Departures'].to_excel("data/"+per+"/dfDepartures_dayData_2018-2019.xlsx")
dfSLC_dayData['EnergyAvg'].to_excel("data/"+per+"/dfEnergyAvg_dayData_2018-2019.xlsx")
dfSLC_dayData['EnergyTot'].to_excel("data/"+per+"/dfEnergyTot_dayData_2018-2019.xlsx")
dfSLC_dayData['Duration'].to_excel("data/"+per+"/dfDuration_dayData_2018-2019.xlsx")
dfSLC_dayData['Charging'].to_excel("data/"+per+"/dfCharging_dayData_2018-2019.xlsx")

#%% Read Day Data 
per = '1hr'

dfSLC_dayData = {};
dfSLC_dayData['Arrivals'] = pd.read_excel("data/"+per+"/dfArrivals_dayData_2018-2019.xlsx", index_col=[0])
dfSLC_dayData['Departures'] = pd.read_excel("data/"+per+"/dfDepartures_dayData_2018-2019.xlsx", index_col=[0])
dfSLC_dayData['EnergyAvg'] = pd.read_excel("data/"+per+"/dfEnergyAvg_dayData_2018-2019.xlsx", index_col=[0])
dfSLC_dayData['EnergyTot'] = pd.read_excel("data/"+per+"/dfEnergyTot_dayData_2018-2019.xlsx", index_col=[0])
dfSLC_dayData['Duration'] = pd.read_excel("data/"+per+"/dfDuration_dayData_2018-2019.xlsx", index_col=[0])
dfSLC_dayData['Charging'] = pd.read_excel("data/"+per+"/dfCharging_dayData_2018-2019.xlsx", index_col=[0])

#%%

def aggData(dfDays, periodsPerDay):
    df = dfDays;
    daysIn = df['Arrivals'].shape[1]
    dfDays_Val = pd.DataFrame(np.zeros((periodsPerDay*daysIn,10)),
              columns=['Hour','DayCnt','DayYr','Arrivals','Departures','Connected','EnergyAvg','EnergyTot','Duration','Charging'])

    r = 0; d = 0; 
    for j in df['Arrivals'].columns:
        print(j)
        dfDays_Val.Hour.iloc[r:r+periodsPerDay] = np.arange(0, periodsPerDay);
        dfDays_Val.DayCnt.iloc[r:r+periodsPerDay] = np.repeat(d, periodsPerDay);
        dfDays_Val.DayYr.iloc[r:r+periodsPerDay] = j;
    
        dfDays_Val.Arrivals[r:r+periodsPerDay] = df['Arrivals'][j];        
        dfDays_Val.Departures[r:r+periodsPerDay] = df['Departures'][j];  
        dfDays_Val.EnergyAvg[r:r+periodsPerDay] = df['EnergyAvg'][j];
        dfDays_Val.EnergyTot[r:r+periodsPerDay] = df['EnergyTot'][j];
        dfDays_Val.Duration[r:r+periodsPerDay] = df['Duration'][j];
        dfDays_Val.Charging[r:r+periodsPerDay] = df['Charging'][j];
    
        d += 1;
        r += periodsPerDay;
    
    # Count connected vehicles
    cnctd = 0;
    for i in range(0,len(dfDays_Val.Arrivals)):
        if dfDays_Val.Hour.at[i] == 0:
            cnctd = 0;
            
        cnctd = cnctd + dfDays_Val.Arrivals.at[i] - dfDays_Val.Departures.at[i];
        
        if cnctd < 0:
            cnctd = 0;
        dfDays_Val.Connected.at[i] = cnctd; 

    return dfDays_Val

ppD = 24;
if per == '1hr':
    ppD = 24;
elif per == '15min':
    ppD = 96;
elif per == '5min_1chgr':
    ppD = 288;

dfSLC_aggData = aggData(dfSLC_dayData, ppD)

# Save
dfSLC_aggData.to_excel("data/"+per+"/dfSLC_aggData_2018-2019.xlsx")

#%% Naive Test-Train Split

#file_aggData = 'data/1hr/dfSLC_aggData_2018-2019.xlsx';
#dfSLC_aggData = pd.read_excel(file_aggData)

# create training and testing vars
def dataSplit(dfSLC_aggData, testPct):
    X_train, X_test = train_test_split(dfSLC_aggData, test_size=testPct, shuffle=False)
    X_train = X_train.reset_index(drop=True);
    X_test = X_test.reset_index(drop=True);
    return X_train, X_test

df_Train_naive, df_Test_naive = dataSplit(dfSLC_aggData, 0.2)

df_Train_naive.to_excel("data/"+per+"/trn_test/dfTrn_Naive.xlsx")
df_Test_naive.to_excel("data/"+per+"/trn_test/dfTest_Naive.xlsx")

#%% Cross Validation Test/Train Data

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

#file_aggData = "data/"+per+"/dfSLC_aggData_2018-2019.xlsx";
#dfSLC_aggData = pd.read_excel(file_aggData)
#dfSLC_aggData = dfSLC_aggData.drop("Idx", axis=1) 

data_Train, data_Test = train_test_split(dfSLC_aggData, test_size=0.2, shuffle=False)

def dataCV(X, folds):
    
    kf = KFold(n_splits=folds) # Define the split - into 5 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    print("--- Training & Validation ---\n",kf) 
    f = 0; X_train={}; X_test={}; 
    
    for train_index, test_index in kf.split(X):
         print("TRAIN:", train_index, "\nTEST :", test_index)
         X_train[f], X_test[f] = X.iloc[train_index], X.iloc[test_index]         
         f += 1;
     
    return X_train, X_test 

X_Train, X_Test = dataCV(data_Train, 5)

#%% Output Data

for i in range(5):
    X_Train[i].to_excel("data/"+per+"/trn_test/x_trn"+str(i)+".xlsx")
    X_Test[i].to_excel("data/"+per+"/trn_test/x_val"+str(i)+".xlsx")

data_Test.to_excel("data/"+per+"/trn_test/y_test.xlsx")
