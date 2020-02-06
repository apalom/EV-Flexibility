# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:55:41 2020

@author: Alex Palomino
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import pymc3 as pm
from pymc3 import model_to_graphviz

#%% Read k-Fold Test-Train Data

df_Train = {}; df_Val = {}; k = 5;
per = "5min_1port";

for i in range(k):

    df_Train[i] = pd.read_excel("data/"+per+"/trn_test/x_trn"+str(i)+".xlsx")#.sample(10*288)    
    df_Val[i] = pd.read_excel("data/"+per+"/trn_test/x_val"+str(i)+".xlsx")#.sample(10*288)
    print(i)
    
#%% PCA 