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
#R-squ. ranges between 0 and 1, with 1 being a perfect fit.

# Initial predicted values
model_fit1 = crab
preds_1 = m1.predict()
model_fit1['preds'] = preds_1.astype(int)
model_fit1['error'] = np.abs(model_fit1['preds'] - model_fit1['Sa'])/model_fit1['Sa']
#model_fit1 = model_fit1.sort_values(by='error')

# The fitted values are shown below. Although the predictor, W, 
#appears to be significant, the fitted values are not close to the true values.

print('\n Poisson(W) Prediction Summary')
print(model_fit1.head(5))

#%% Dealing with Overdispersion
#As discussed in the above discussion, there can be an issue with overdispersion. 
#Then, it may be more appropriate to fit a negative binomial model in this case.

m_nb = negativebinomial('Sa ~ W', data = crab).fit()
print(m_nb.summary())

# Initial predicted values
model_fitNB = crab
preds_NB = m_nb.predict()
model_fitNB['preds'] = preds_NB.astype(int)
model_fitNB['error'] = np.abs(model_fitNB['preds'] - model_fitNB['Sa'])/model_fitNB['Sa']
#model_fitNB = model_fitNB.sort_values(by='error')

print('\n NegBinomial(W) Prediction Summary')
print(model_fitNB.head(5))

#%% Adding Another Predictor “Crab Color”-“C”
#Besides the impact of overdispersion, there are some other influential factors, 
#such as the lack of predictors. We want to check whether the lack of fit is caused 
#by the lack of the number of predictors. Then, the variable, C, is added to the 
#model.

col_dummies = pd.get_dummies(crab['C']).rename(columns = lambda x: 'Col' + str(x))
dataWithDummies = pd.concat([crab, col_dummies], axis = 1) 
dataWithDummies.drop(['C', 'Col1'], inplace = True, axis = 1)
dataWithDummies = dataWithDummies.applymap(np.int)
dataWithDummies.head(5)

print('Data with Dummies \n ', dataWithDummies.head(5), '\n')

#Let color type 1 be the reference level. Compute the design matrix, X and put 
#the response aside, then store it into Y.

columns = ['W', 'Col2', 'Col3', 'Col4']
X = [elem for elem in dataWithDummies[columns].values]
X = sm.add_constant(X, prepend = False)
Y = [elem for elem in dataWithDummies['Sa'].values]

#Fit a poisson model using X and Y. Then summarize the model.

m2 = sm.Poisson(Y, X).fit()
print(m2.summary())

# Initial predicted values
model_fit2 = crab
preds_2 = m1.predict()
model_fit2['preds'] = preds_2.astype(int)
model_fit2['error'] = np.abs(model_fit2['preds'] - model_fit2['Sa'])/model_fit2['Sa']

#From the output, LLR p value is still significant, which indicates that adding 
#another predictor crab color doesn’t improve goodness of fit for the model.

#We can also plot the true values of the response vs the fitted values to see 
#the goodness of the fit.
preds = m2.predict()
plt.plot(range(len(Y)), Y, 'r*-', range(len(Y)), preds, 'bo-')
plt.title('True Sa vs fitted values')
plt.legend(['Real Values', 'Fitted Values'])
plt.show()

#%% Adding Another Predictor “Crab Weight”-"Wt"

m2a = poisson('Sa ~ Wt', data = crab).fit()
print(m2a.summary())

# Initial predicted values
model_fit2a = crab
preds_2a = m2a.predict()
model_fit2a['preds'] = preds_2a.astype(int)
model_fit2a['error'] = np.abs(model_fit2['preds'] - model_fit2a['Sa'])/model_fit2a['Sa']
#model_fit2a = model_fit2a.sort_values(by='error')

# The fitted values are shown below. Although the predictor, W, 
#appears to be significant, the fitted values are not close to the true values.

print('\n Poisson(W) Prediction Summary')
print(model_fit2a.head(5))

