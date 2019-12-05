# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:32:56 2019

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

# Import Data
filePath = 'data/Session-Details-Summary-20191203.csv';
dataRaw = pd.read_csv(filePath);

dataRawHead = dataRaw.head(100);
#dataTypes = data.dtypes;

allColumns = list(dataRaw);

#%% Dataframe Preparation

def filterPrep(df, string, fltr):

    colNames = ['EVSE ID', 'Port Number', 'Port Type', 'Station Name', 'Plug In Event Id', 'City', 'Latitude', 'Longitude',
                'User ID', 'Driver Postal Code', 'Start Date', 'End Date', 'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)',
                'Energy (kWh)', 'Ended By', 'Start SOC', 'End SOC'];

    df = pd.DataFrame(df, index=np.arange(len(df)), columns=colNames)

    df = df.loc[df['Port Type'] == 'DC Fast']

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
    yr = 2017
    df = df.loc[(df['Start Date'] > datetime.date(yr,12,1)) & (df['Start Date'] < datetime.date(yr+2,12,1))]

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
    #df['isWeekday'] = df['DayofWk'].apply(lambda x: 1 if x <=4 else 0)
    #df = df.loc[df['isWeekday'] == 1]
    df['Year'] = df['Start Date'].apply(lambda x: x.year)
    df['StartHr'] = df['Start Date'].apply(lambda x: x.hour + x.minute/60)
    #df['StartHr'] = df['StartHr'].apply(lambda x: np.floor(x))
    df['StartHr'] = df['StartHr'].apply(lambda x: round(x * 4) / 4)
    df['EndHr'] = df['End Date'].apply(lambda x: x.hour + x.minute/60)
    #df['EndHr'] = df['EndHr'].apply(lambda x: np.floor(x))
    df['EndHr'] = df['EndHr'].apply(lambda x: round(x * 4) / 4)
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

    daysTot =  (df['Start Date'].iloc[len(df)-1] - df['Start Date'].iloc[0]).days+1

    return df, daysTot;

# Salt Lake City Sessions
dfUtah, daysTot = filterPrep(dataRaw, "Salt Lake City", False)

# Save
dfUtah.to_excel("data/DCFC_dfUtah_Alldays_2018-2019.xlsx")

#%% Training and Testing for a Single Day

import random

def testTrain(df, day, p):

    #df = df.loc[df.DayofWk == day]
    df = df.reset_index(drop=True)
    daysIn = list(set(list(df.DayofYr)))
    daysIn.sort()

    # Define list of days for training and testing
    daysTrain = random.sample(daysIn, int(p*len(daysIn)))
    daysTest = list(set(daysIn) - set(daysTrain))

    # Sample training data
    dfTrain = df.loc[df.DayofYr.isin(daysTrain)]
    dfTest = df.loc[df.DayofYr.isin(daysTest)]

    dfTrain = dfTrain.sort_values(by=['dayCount'])
    dfTest = dfTest.sort_values(by=['dayCount'])

    return dfTrain, dfTest

# Inputs (dfAll, Day of Week [Mon = 0, Sat = 5] ,percent Training Data)
dfTrain15, dfTest15 = testTrain(dfSLC_15min, 0, 0.20)

#dfTrain15.to_csv('data\dfTrain_all15.csv')
#dfTest15.to_csv('data\dfTest_all15.csv')

#daysInTrn = len(list(set(list(dfTrain.DayofYr))))

#%% Calculate Connected EVs per Day/Hr and Calculate Mean, 1st and 2nd Standard Deviation of Connected Vehicles

def quants(df, weekday):

    dctQuant = {}; dctDay = {};

    daysIn = list(set(df.dayCount))
    daysIn.sort()

    dfArrivals = pd.DataFrame(np.zeros((4*24,len(set(df.dayCount)))),
                          index=np.arange(0,24,0.25), columns=daysIn)

    dfEnergy = pd.DataFrame(np.zeros((4*24,len(set(df.dayCount)))),
                          index=np.arange(0,24,0.25), columns=daysIn)

    dfDuration = pd.DataFrame(np.zeros((4*24,len(set(df.dayCount)))),
                          index=np.arange(0,24,0.25), columns=daysIn)

    dfCharging = pd.DataFrame(np.zeros((4*24,len(set(df.dayCount)))),
                          index=np.arange(0,24,0.25), columns=daysIn)

    for d in df.dayCount:
        print('Day: ', d)
        dfDay = df[df.dayCount == d]
        cnct = dfDay.StartHr.value_counts()
        cnct = cnct.sort_index()

        energy = dfDay['Energy (kWh)'].groupby(dfDay.StartHr).sum()
        duration = dfDay['Duration (h)'].groupby(dfDay.StartHr).mean()
        charging = dfDay['Charging (h)'].groupby(dfDay.StartHr).mean()

        dfArrivals.loc[:,d] = cnct
        dfArrivals.loc[:,d] = np.nan_to_num(dfArrivals.loc[:,d])

        dfEnergy.loc[:,d] = energy
        dfEnergy.loc[:,d] = np.nan_to_num(dfEnergy.loc[:,d])

        dfDuration.loc[:,d] = duration
        dfDuration.loc[:,d] = np.nan_to_num(dfDuration.loc[:,d])

        dfCharging.loc[:,d] = charging
        dfCharging.loc[:,d] = np.nan_to_num(dfCharging.loc[:,d])

#        quants = pd.DataFrame(np.zeros((24,6)),
#                            index= np.arange(0,24,1),
#                            columns=['-2_sigma','-1_sigma','mu','+1_sigma','+2_sigma','stddev'])
#        quants['-2_sigma'] = dfDays.quantile(q=0.023, axis=1)
#        quants['-1_sigma'] = dfDays.quantile(q=0.159, axis=1)
#        quants['mu'] = dfDays.quantile(q=0.50, axis=1)
#        quants['+1_sigma'] = dfDays.quantile(q=0.841, axis=1)
#        quants['+2_sigma'] = dfDays.quantile(q=0.977, axis=1)
#        quants['stddev'] = np.std(dfDays, axis=1)
#        dctQuant[dfNames[i]] = quants;
    dctDay['Arrivals'] = dfArrivals;
    dctDay['Energy'] = dfEnergy;
    dctDay['Duration'] = dfDuration;
    dctDay['Charging'] = dfCharging;

    return dctDay, dctQuant

dfDays_Test15, dfQuants = quants(dfTest15, True)

#%% Create Hour_DayCnt_DayYr_Connected data

#dfHrCnctd = {};
df = dfDays_Test15

daysIn = df['Arrivals'].shape[1]
dfDays_Val = pd.DataFrame(np.zeros((4*24*daysIn,7)),
              columns=['Hour','DayCnt','DayYr','Arrivals','Energy','Duration','Charging'])

r = 0; d = 0;

for j in df['Arrivals'].columns:
    print(j)
    dfDays_Val.Hour.iloc[r:r+4*24] = np.arange(0,24,0.25);
    dfDays_Val.DayCnt.iloc[r:r+4*24] = np.repeat(d, 4*24);
    dfDays_Val.DayYr.iloc[r:r+4*24] = j;

    dfDays_Val.Arrivals[r:r+4*24] = df['Arrivals'][j];
    dfDays_Val.Energy[r:r+4*24] = df['Energy'][j];
    dfDays_Val.Duration[r:r+4*24] = df['Duration'][j];
    dfDays_Val.Charging[r:r+4*24] = df['Charging'][j];

    d += 1;
    r += 4*24;

dfDays_Test15Val = dfDays_Val
dfDays_Test15Val.to_csv('data\dfDays_Test15Val.csv')
#dfHrCnctd['Train'].to_csv('data\hdc_wkdy_TRAIN.csv')

#%% Create Fitting Data

def dfFitting(dfs, dfDays):

    dfFits = {}
    i = 0;
    for frame in dfs:

        dlen = 24*len(set(dfs[i].dayCount))
        dfDayCol = pd.DataFrame(np.zeros((dlen,6)), columns=['Hour','isWeekday','DayWk','DayYr','DayCnt','Connected'])
        dayList = list(dfDays[i]);

        for c in np.arange(0,dfDays[i].shape[1]):
            print('df: ', i, ' | Day: ', c);
            isWkdy = 0;

            # Day of Week [Mon = 0, Sat = 5]
            if np.mod(dayList[c],7) < 5:
                isWkdy = 1;

            for h in np.arange(0,24):
                #print('  Hr: ', h);
                dfDayCol.Hour.at[(c*24)+h] = int(h);
                dfDayCol.isWeekday.at[(c*24)+h] = isWkdy;
                dfDayCol.DayWk.at[(c*24)+h] = np.mod(dayList[c],7);
                dfDayCol.DayYr.at[(c*24)+h] = dayList[c];
                dfDayCol.DayCnt.at[(c*24)+h] = c;
                dfDayCol.Connected.at[(c*24)+h] = dfDays[i].iloc[h,c];

        dfFits[i] = dfDayCol;
        i += 1;

    return dfFits

dfFits = dfFitting([dfTrain, dfTest], dfDays)

dfFits[0].to_excel('data\dfFitsTrain_all.xlsx')
dfFits[1].to_excel('data\dfFitsTest_all.xlsx')

#%% Scatterplot Fitting Data

import seaborn as sns

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(16,6))
fig.tight_layout()

endHr = 143
ax = sns.stripplot(x=dfFitTrain.head(endHr).index, y="Connected", hue="Day",
                   s=5, data=dfFitTrain.head(endHr), jitter=True)

ax.set(xlabel='Hour',  ylabel='EV Connected', title='Training Data - Monday',
       xticks=np.arange(0,endHr,4), xticklabels = np.arange(0,endHr,4), ylim=((0,16)))

ax.legend_.remove()
sns.despine()

#%% GP Fit

# Import Kernels
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.5),
           1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0)),
           ConstantKernel(0.1, (0.01, 10.0))
               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
           1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)]

kernel = (5**2 * RBF(length_scale=24, length_scale_bounds=(0.1, 48))
            + 2**2 * RationalQuadratic(length_scale=1.0, alpha=0.1))

days = 10;

# Training Data
X = np.array(dfFitTrain.head(24*days).index.values).reshape(-1, 1)
y = np.array(dfFitTrain.head(24*days).Connected)

# Fit GP to Training Data
gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              optimizer=None, normalize_y=True)

# GPML Kernel on Prior (Initial Hyperparameters)
gp.fit(X, np.log(y))

print("\n GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

# Learned Kernel
gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              optimizer='fmin_l_bfgs_b', normalize_y=True)

# GPML Kernel on Prior (Optimized Hyperparameters)
gp.fit(X, np.log(y))

print("\n Learned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

#%% Plot

X_ = np.linspace(X.min(), X.max() - 24, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

# Illustration
plt.figure(figsize=(16, 8))
plt.scatter(X, y, c='k')
plt.plot(X_, y_pred)
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.5, color='k')
plt.xticks(np.arange(0,264,12))
plt.xlim(X_.min(), X_.max())
plt.xlabel("Year")
plt.ylim(0,15)
plt.ylabel(r"EVs Connected")
plt.title(r"Monday Training Data")
plt.tight_layout()
plt.show()

#%%

# Plot GP Prior
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(np.arange(0,24), y_mean, 'k', lw=2, zorder=9)
plt.fill_between(np.arange(0,24), y_mean - y_std, y_mean + y_std,
                 alpha=0.2, color='k')
plt.xticks(np.arange(0,26,2))

nS = 1
y_samples = gp.sample_y(Xc, nS)

y_samples1 = np.zeros((len(y_samples),nS))

for j in range(len(y_samples1)):
    y_samples1[j] = y_samples[j]

Xc1 = np.repeat(Xc, nS, axis=1)

plt.plot(Xc1, y_samples1, lw=1)

plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

gp.kernel_

## Testing Data Predicted from GP
#x_pred = np.array(dfFitTest.Hour).reshape(-1, 1)
#y_pred, sigma = gp.predict(x_pred, return_std=True)

#%%

dfMonday = dfSLC.loc[dfSLC.DayofWk >= 5]
dfMonday = dfMonday.reset_index(drop=True)

#%% Remove data outside of 2nd standard deviation

#dim = 'Duration (h)'
dim = 'Energy (kWh)'
stdDev2 = np.round((dfSLC[dim].quantile(q=0.977)),0)

dfCln = dfSLC.loc[dfSLC[dim] < stdDev2]

agg_mu = np.mean(dfCln[dim].values)
agg_sigma = np.std(dfCln[dim].values)

# Sturge’s Rule for Bin Count
kBins = 1 + 3.22*np.log(len(dfCln))
print('Number of Bins: ', int(kBins))
# results approximately in 1 kWh wide bins for Energy

plt.style.use('ggplot')
plt.figure(figsize=(12,6))

font = {'family': 'Times New Roman', 'weight': 'light', 'size': 16}
plt.rc('font', **font)

dfCln[dim].plot.hist(grid=True, bins=np.arange(0,(kBins)),
                     density=True, rwidth=0.9, color='#607c8e')


plt.xlabel(dim)
plt.ylabel('Density')

#%% Hourly Plot Histograms

hrs = np.arange(0,24)
hists = {}

fig, axs = plt.subplots(4, 6, figsize=(16,12), sharex=True, sharey=True)
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
plt.rc('font', **font)

r,c = 0,0;

for hr in hrs:
    mask = (dfCln['StartHr'] == hr)
    df_hr = dfCln[mask]

    if len(df_hr) > 0:
        kBins = 1 + 3.22*np.log(len(df_hr)) #Sturge's Rule for Bin Count
        hists[hr] = np.histogram(df_hr[dim], bins=int(kBins))
    else:
        hists[hr] = np.histogram(0)

    print('position', r, c)
    axs[r,c].hist(df_hr[dim], edgecolor='white', color='#E3A79D', linewidth=0.5, bins=int(kBins), density=True)
    axs[r,c].set_title('Hr: ' + str(hr))
    axs[r,c].text(9, 0.35,  str(len(df_hr)) + ' samples')#, ha='center', va='center',)
    #axs[r,c].set_xlim(0,22)
    #axs[r,c].set_xticks(np.arange(0,22+4,4))

    # Subplot Spacing
    c += 1
    if c >= 6:
        r += 1;
        c = 0;
        if r >= 4:
            r=0;


fig.text(0.5, 0.0, 'Energy (kWh)', ha='center')
fig.text(0.0, 0.5, 'Density', va='center', rotation='vertical')

#fig.suptitle('Hourly Histogram: '+ dim, y = 1.02)
xM, bS = int(np.max(dfCln[dim])), 4
#plt.xlim(0,xM)
#plt.xlim(0,22)
plt.xticks(np.arange(0,xM+bS,bS))
plt.ylim(0,0.4)

fig.tight_layout()
plt.show()


#%% Plot Fits

import scipy
import seaborn as sns
from scipy import stats
plt.rcParams['text.usetex'] = True

#sns.set_color_codes()
#sns.set_style('darkgrid')
plt.figure(figsize=(10,8))
x = dfCln[dim]
#xN = (x - agg_mu)/agg_sigma

fit_q = stats.kstest(x, 'norm', args=stats.norm.fit(x), N=1000)
ax = sns.distplot(x, fit=stats.norm, kde=False,
                  fit_kws={'color':'blue', 'label':'norm: $\chi^2$ =${0:.2g}'.format(resultsChi.loc['norm']['chi_square'])})
print('norm: ', fit_q)

fit_q = stats.kstest(x, 'gamma', args=stats.gamma.fit(x), N=1000)
ax = sns.distplot(x, fit=stats.gamma, hist=False, kde=False,
                  fit_kws={'color':'green', 'label':'gamma: $\chi^2$ =${0:.2g}'.format(resultsChi.loc['gamma']['chi_square'])})
print('gamma: ', fit_q)

fit_q = stats.kstest(x, 'beta', args=stats.beta.fit(x), N=1000)
ax = sns.distplot(x, fit=stats.beta, hist=False, kde=False,
                  fit_kws={'color':'red', 'label':'beta: $\chi^2$ =${0:.2g}'.format(resultsChi.loc['beta']['chi_square'])})
print('beta: ', fit_q)


fit_q = stats.kstest(x, 'expon', args=stats.expon.fit(x), N=1000)
ax = sns.distplot(x, fit=stats.expon, hist=False, kde=False,
                  fit_kws={'color':'grey', 'label':'expon: $\chi^2$ =${0:.2g}'.format(resultsChi.loc['expon']['chi_square'])})
print('expon: ', fit_q)

ax.legend()

# Gamma Params Duration: α, loc, β
params_gamma = stats.gamma.fit(x);

# Beta Params Energy: α, β, loc (lower limit), scale (upper limit - lower limit)
params_beta = stats.beta.fit(x);

# Beta Params Energy: α, loc, scale
params_skewnorm = stats.skewnorm.fit(x);

#%% Test Chi-Square
# https://pythonhealthcare.org/2018/05/03/81-distribution-fitting-to-data/

import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt
%matplotlib inline
# Load data and select first column

from sklearn import datasets
y=dfCln[dim].values

# Create an index array (x) for data
x = np.arange(len(y))
size = len(y)

#plt.hist(y)
#plt.show()

sc=StandardScaler()
yy = y.reshape (-1,1)
sc.fit(yy)
y_std =sc.transform(yy)
y_std = y_std.flatten()
y_std
del yy

# Turn off code warnings (this is not recommended for routine use)
import warnings
warnings.filterwarnings("ignore")

# Set up list of candidate distributions to use
# See https://docs.scipy.org/doc/scipy/reference/stats.html for more

dist_names = ['beta',
              'expon',
              'gamma',
              'norm']

# Set up empty lists to stroe results
chi_square = []
p_values = []

# Set up 50 bins for chi-square test
# Observed data will be approximately evenly distrubuted aross all bins
percentile_bins = np.linspace(0,100,51)
percentile_cutoffs = np.percentile(y_std, percentile_bins)
observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
cum_observed_frequency = np.cumsum(observed_frequency)

# Loop through candidate distributions

for distribution in dist_names:
    # Set up distribution and get fitted distribution parameters
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)

    # Obtain the KS test P statistic, round it to 5 decimal places
    p = scipy.stats.kstest(y_std, distribution, args=param)[1]
    p = np.around(p, 5)
    p_values.append(p)

    # Get expected counts in percentile bins
    # This is based on a 'cumulative distrubution function' (cdf)
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                          scale=param[-1])
    expected_frequency = []
    for bin in range(len(percentile_bins)-1):
        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)

    # calculate chi-squared
    expected_frequency = np.array(expected_frequency) * size
    cum_expected_frequency = np.cumsum(expected_frequency)
    ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
    chi_square.append(ss)

# Collate results and sort by goodness of fit (best at top)

resultsChi = pd.DataFrame()
resultsChi['Distribution'] = dist_names
resultsChi['chi_square'] = chi_square
resultsChi['p_value'] = p_values
resultsChi.sort_values(['chi_square'], inplace=True)
resultsChi = resultsChi.set_index('Distribution')

# Report results

print ('\nDistributions sorted by goodness of fit:')
print ('----------------------------------------')
print (resultsChi)

#%% Margin Plots

import seaborn as sns
from scipy import stats

yM, cc, ttl = 20, 'lightblue', 'dfWknd-Cln'
g = sns.jointplot(dfCln.StartHr, dfCln[dim], color=cc, kind='kde')
g.ax_joint.set_xticks(np.arange(0,26,2))
#g.ax_joint.set_ylim(0,yM)
g.ax_joint.set_title(ttl, x = 1.1, y = 1.0)
#g.annotate(stats.pearsonr, loc=(1.2,1), fontsize=0.1)

#%% Import Training Connected Dat

# Raw Data
filePath = 'data/train_connected_per_hr.csv';

# Import Data
dfCnctd = pd.read_csv(filePath);
dfCnctdT = dfCnctd.transpose();


#%% Normality Tests

print(dim)

#Perform the Shapiro-Wilk test for normality.
stat, p = stats.shapiro(x)

alpha = 0.05
print('\n')
if p > alpha:
    print('Shapiro: Sample looks Gaussian (fail to reject H0)')
else:
    print('Shapiro: Sample does not look Gaussian (reject H0)')
print(stat,p)

# Perform Anderson-Darling test for normality
result = stats.anderson(x, dist='norm')
stat = round(result.statistic, 4)
print('\n')
p = 0
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('Anderson: Sample looks Gaussian (fail to reject H0)')
    else:
        print('Anderson: Sample does not look Gaussian (reject H0)')
    print('\t', len(x), stat, sl, cv)

# Perform K-S test for normality
distr = getattr(stats, 'norm')
params = distr.fit(x)
stat, p = stats.kstest(x, 'norm', args=params, N=1000)
print('\n')
if p > alpha:
    print('K-S: Sample looks Gaussian (fail to reject H0)')
else:
    print('K-S: Sample does not look Gaussian (reject H0)')
print(stat,p)


#%%

ax = sns.distplot(dfMon[dim], rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 1, "label": "KDE"},
                   hist_kws={"histtype": "step", "linewidth": 1.5,
                             "alpha": 1, "color": "g"})

#%% Plot Connected Quants

quantData.plot()

plt.xlabel('Time (hr)')
plt.ylabel('Count')
plt.title('EV Sessions Started, SLC-Weekday')
plt.xlim(0,24)
plt.xticks(np.arange(0,26,2))

#%% Create a Sessions Plot

from matplotlib import pyplot as plt
plt.figure(figsize=(10,8))

i = 0;
dfSessions = pd.DataFrame(columns=['StartHr','Charging','Duration','Energy'])

for idx,row in dfSLC.sample(2000).iterrows():
    i += 1;
    dfSessions.at[i] = [row.StartHr, row['Charging (h)'], row['Duration (h)'], row['Energy (kWh)']]
    plt.plot((row.StartHr,row.EndHr),(row['Energy (kWh)'], row['Energy (kWh)']),
             linewidth=0.5, alpha=0.66)

plt.xlabel('Time (hr)')
plt.ylabel('Energy (kWh)')
plt.title('EV Sessions Energy, SLC-Weekday')
plt.xlim(0,24)
plt.xticks(np.arange(0,26,2))

#%% Poisson Distribution Fit

from scipy.stats import poisson

N = 21;

poissMatch = np.zeros((N,3))

for n in range(1, N):

    mean, var, skew, kurt = poisson.stats(n, moments='mvsk')
    print(mean, var)
    poissMatch[n,0] = mean
    poissMatch[n,1] = mean + var
    poissMatch[n,2] = mean - var

#%% Poisson Distribution Fit

trials = 20000
N = 21


poissMatch = pd.DataFrame(np.zeros((3*N,3)), columns=['mu','val','cat'])
poissTrial = pd.DataFrame(np.zeros((trials,2)), columns=['mu','val'])

for n in range(0,N):

    poisRdm = np.random.poisson(n, trials)
    poissTrial.iloc[n:trials+(n)] = [np.array(trials*([n]-1)), poisRdm]

    mu = np.mean(poisRdm)
    poissMatch.at[n] = [mu, mu, 'mu']

    var = np.var(poisRdm)
    poissMatch.at[n+N] = [mu, mu-var, '-var']

    var = np.var(poisRdm)
    poissMatch.at[n+2*N] = [mu, mu+var, '+var']


#%% Plot Poisson Distribution

import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.0})

colors = ["black", "grey", "grey", ]

# Show the results of a linear regression within each dataset
lm = sns.lmplot(x="mu", y="val", hue="cat", data=poissMatch, scatter=True,
           palette=sns.xkcd_palette(colors), scatter_kws={"s": 25, "alpha": 1})

lm.axes[0,0].set_xlim(0,)

#%%

evCount = dfWkdy.values.flatten()

#%%

xD = np.mean(dfWkdy,axis=1)
yD = np.var(dfWkdy,axis=1)

plt.scatter(xD, yD)
plt.xlim(0,16)
plt.xlim(0,16)

#%%

time = np.zeros((21,1))
time = np.arange(0,21)

idxMat = np.full((20000,21), time)

#idx1 = np.repeat(time, N, axis=1)

plt.scatter(x=idxMat,y=dfPoissTrial.values)

#%%

dfPoissTrial = pd.DataFrame(data=poissTrial)
sns.regplot(x=dfPoissTrial.index,y=dfPoissTrial.values,data=dfPoissTrial)

#%%

from scipy.stats import nbinom
import matplotlib.pyplot as plt

result_NB_20k = pd.read_xls('results/NB_20kpool.xlsx');

result_NB_20k['r'], result_NB_20k['p'] = convert_params(result_NB_20k['alpha'],result_NB_20k['mu'])

#%%
from scipy.stats import nbinom
import matplotlib.pyplot as plt

x = np.arange(0, 17, 1)

out_r, out_p = convert_params(out_mu, out_alpha)

for i in np.arange(0,24,6):

    r = out_r[i]; p = out_p[i];
    plt.plot(x, nbinom.pmf(x, r, p), label=i)

plt.xlim((0,10))
plt.legend()
#ax.vlines(x, 0, nbinom.pmf(x, r, p), colors='b', lw=5, alpha=0.5)

#%%

rdmNB = nbinom.rvs(r, p, size=10000)
plt.hist(rdmNB, density=True)

#%%

def convert_params(mu, alpha):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    Parameters
    ----------
    mu : float
       Mean of NB distribution.
    alpha : float
       Overdispersion parameter used for variance calculation.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    var = mu + alpha * mu ** 2
    p = (var - mu) / var
    r = mu ** 2 / (var - mu)
    return r, p


#%% DCFC Start/End SOC

import seaborn as sns

#sns.set_style("whitegrid")

plt.style.use('ggplot')
plt.figure(figsize=(12,6))

font = {'family': 'Times New Roman',
        'weight': 'light',
        'size': 16}
plt.rc('font', **font)

b = np.arange(0,70,5)
plt.hist(dfUtah['Energy (kWh)'], bins=b, density=False, color='g',
                lw=2, alpha=1, histtype='step', label='kWh')
#plt.hist(dfUtah['Charging (h)'], bins=b, density=False,
#                lw=2, alpha=1, histtype='step', label='Charging Time')

plt.title('Utah DCFC (2018-2019)')
plt.ylabel('Frequency')
plt.xlabel('Energy')
#plt.xlim((0,4))
plt.legend()
