# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:55:41 2020

@author: Alex Palomino
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#%% Read-In Data

dfSLC_aggData = pd.read_excel("data/"+per+"/dfSLC_aggData_2017-2019_1port.xlsx")    
data_x = dfSLC_aggData[['Hour','DayWk']] 
data_lbl = dfSLC_aggData[['Connected']] 

#%% Test/Train Split Data

train_data, test_data, train_lbl, test_lbl = train_test_split(data_x, data_lbl, test_size=0.2, shuffle=False)    
    
train_lbl = train_lbl.Connected
test_lbl = test_lbl.Connected
#%% Standardize the Data

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_data)
# Apply transform to both the training set and the test set.
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

#%% Apply Logistic Regression to the Transformed Data

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(max_iter=1000)

# Training the model on the data, storing the information learned from the data
logisticRegr.fit(train_data, train_lbl)

# Predict for Multiple Observations (images)
test_pred = logisticRegr.predict(test_data)

#% Measure Model Performance
score = logisticRegr.score(test_data, test_lbl)
print(score)
