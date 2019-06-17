# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:02:56 2019

@author: Alex
https://jbhender.github.io/Stats506/F17/Projects/Poisson_Regression.html

The study investigated factors that affect whether the female crab had 
any other males, called satellites, residing nearby her. Explanatory 
variables thought possibly to affect this included the female crab’s:
    color (C), spine condition (S) , weight (Wt), and carapace width (W). 
    
The response outcome for each female crab is her number of satellites (Sa).

"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import numpy as np
import matplotlib.pyplot as plt
import math

crab = pd.read_csv("crab.csv", names = ['Obs', 'C', 'S', 'W', 'Wt', 'Sa'])

crab.head(5)
crab.describe()

#%% compute the average numbers of satellites by color type and 
#the result seems to suggest that color type is a good candidate 
#for predicting the number of satellites, our outcome variable

uniqCols = sorted(crab['C'].unique())

mn_c = []
for elem in uniqCols:
    mn_c.append(crab[crab['C'] == elem]['Sa'].mean())
    
std_c = []
for elem in uniqCols:
    std_c.append(crab[crab['C'] == elem]['Sa'].std())
    
mn_c = np.array(mn_c)
std_c = np.array(std_c)
mn_std = pd.DataFrame(mn_c, columns = ["mean"])
mn_std["SD"] = std_c

print(mn_std)

#%%  plot a conditional histogram separated out by color type 
#to show the distribution

histData = []
for elem in uniqCols:
    histData.append(crab[crab['C'] == elem]['Sa'].values)
    
plt.hist(tuple(histData), bins = 10, normed = True, histtype = 'bar', label = map(lambda x: 'Color' + str(x), uniqCols))
plt.hist(crab.Sa, bins = 10, normed = True, histtype = 'step', color='grey', label='totals')
plt.legend()
plt.ylabel('Count')
plt.title('Histogram for each color')
plt.show()

#%% Now, we are ready to perform our Poisson model analysis using the 
#poisson function imported from the package, statsmodels.formula.api.
#First, regress the response, Sa, on the two continuous predictors, 
#W and Wt. Store this model in the object m1 and then summarize the model.
#LLR Test = Log-Likelihood Ratio Test (p-value)

m1 = poisson('Sa ~ W', data = crab).fit()
print(m1.summary())

# Initial predicted values
model_fit1 = crab
preds_1 = m1.predict()
model_fit1['preds'] = preds_1

print('\n Prediction Summary')
print(model_fit1.head(5))