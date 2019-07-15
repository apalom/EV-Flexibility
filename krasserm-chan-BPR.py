# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:58:48 2019

@author: Alex

http://krasserm.github.io/2018/03/19/gaussian-processes/
Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning.
Bayesian Poisson Regression for Crowd Counting [Chan]
"""

import numpy as np

# Kernel Definition
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    ''' Isotropic squared exponential kernel (RBF). Computes a covariance matrix from points 
    in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). 
    Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


from numpy.linalg import inv
#from scipy.sparse import csc_matrix
#from scipy.sparse.linalg import inv

# Posterior Prediction
def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0): #, sigma_y=1e-8):
    ''' Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s. 
    Args: X_s: New input locations (n x d). X_train: Training locations (m x d). 
    Y_train: Training targets (m x 1). l: Kernel length parameter. 
    sigma_f: Kernel vertical variation parameter. sigma_y: Noise parameter. 
    Returns: Posterior mean vector (n x d) and covariance matrix (n x n). '''
    yNoise = 1/Y_train
    yNoise[yNoise == np.inf] = 0
    sigY = np.eye(len(yNoise))*(yNoise)
    #sigY[sigY == -0.0] = 0.0    

    K = kernel(X_train, X_train, l, sigma_f)# + sigY**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
        
    # Equation (Kr4) to (Chan47)
    K_sigY = K + sigY 
    K_sigY[K_sigY < 1e-4] = 0
    #K_sigY = csc_matrix(K_sigY)
    mu_s = K_s.T.dot(inv(K_sigY)).dot(np.log(Y_train))

    # Equation (Kr5) to (Chan48)
    cov_s = K_ss - K_s.T.dot(inv(K + sigY)).dot(K_s)
    
    return mu_s, cov_s, sigY

# Krasserm Bayesian ML GP Utility
# https://github.com/krasserm/bayesian-machine-learning/blob/af6882305d9d65dbbf60fd29b117697ef250d4aa/gaussian_processes_util.py#L7
    
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    
    plt.subplots(figsize=(10,6))
    
    X = X.ravel()
    mu = mu.ravel()
    # 1.96 is the approximate value of the 97.5 percentile point of the normal distribution. 
    # 95% of the area under a normal curve lies within roughly 1.96 standard deviations of the mean, 
    # and due to the central limit theorem, this number is therefore used in the construction of 
    # approximate 95% confidence intervals
    
    #uncertainty = 1.96 * np.sqrt(np.diag(cov))
    uncertainty = 1.96 * np.sqrt(cov)
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.75, ls=':', label=f'Sample {i+1}')
        #plt.scatter(X, sample, s = 7, label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)

#%% --- Split Zero from Count Data
    
dfCount = dfFitTrain.loc[dfFitTrain.Connected>0]
dfZeros = dfFitTrain.loc[dfFitTrain.Connected==0]

#%% --- Fit Continuous Distribution to Mean 
#https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
    
import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Fit to Mean
x = np.arange(0, 24, 1).reshape(-1, 1)
mu_act = np.array(quantDataTrain.mu.values).reshape(-1,1)

# Fit to All Data
x_scatter = np.array(dfFitTrain.Hour).reshape(-1, 1)
y_scatter = np.array(dfFitTrain.Connected).reshape(-1,1)

polynomial_features = PolynomialFeatures(degree=9)
x_poly = polynomial_features.fit_transform(x)

# Fit to polynomial linear regression
model = LinearRegression()
model.fit(x_poly, mu_act)
y_poly_pred = (model.predict(x_poly))
#y_poly_pred[y_poly_pred < 0] = 0.0

for yy in range(len(y_poly_pred)):
    if y_poly_pred[yy][0] < 0:
        y_poly_pred[yy][0] = 0;

rmse = np.sqrt(mean_squared_error(mu_act,y_poly_pred))
r2 = r2_score(mu_act,y_poly_pred)
print('Root Mean Squared Error: {0:.3f}'.format(rmse))
print('R-square: {0:.3f}'.format(r2))

#%% Plot Continuous Distribution - Polynomial Regression Predicted Mean

def jitter(x, y):
    
    rx = np.random.rand(len(x));
    posneg = (2*np.random.randint(0,2,size=(len(x)))-1);
    #x = [rx*posneg][0] + x.reshape(1,-1)[0];
    
    ry = np.random.rand(len(y));
    posneg = (2*np.random.randint(0,2,size=(len(y)))-1);
    y = [ry*posneg][0] + y.reshape(1,-1)[0];
    
    return x, y;

import seaborn as sns

plt.figure(figsize=(8,12))
#sns.set(style="whitegrid")

plt.subplot(2, 1, 1)
ax = sns.swarmplot(x='Hour', y='Connected', data=dfFitTrain)  
ax.set(xticklabels=[], xlabel='')
ax.set_title('Monday Data - Jitter')

plt.subplot(2, 1, 2)
x_jitter, y_jitter = jitter(x_scatter, y_scatter);
plt.scatter(x_scatter, y_jitter, s=2, color='grey', label='data')
plt.scatter(x, mu_act, s=20, color='k', label='mu_act')
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.ylabel('Connected')
plt.xlabel('Hour of Day')
plt.xticks(np.arange(0,24))
plt.plot(x, y_poly_pred, color='purple', label='mu_poly_predicted')
plt.title('Monday - Polynomial Predicted Mean')
plt.legend()
plt.grid(False)
plt.show()

#%% --- Prior ---

# Finite number of points (12)
Xpr = np.arange(0, 24, 1).reshape(-1, 1)

# Mean and covariance of the prior
mu_act = np.array(quantDataTrain.mu.values).reshape(-1,1)
cov = kernel(Xpr, Xpr)

# Mean calculated by linear regression model fit
Xpr_poly = polynomial_features.fit_transform(Xpr)
mu_pr = (model.predict(Xpr_poly))
mu_pr[mu_pr < 0.0] = 0.0

# Draw three samples from the prior
#samples = np.random.multivariate_normal(mu.ravel(), cov, 3)
samples = np.random.poisson(mu_pr.ravel(), size = (4,len(Xpr)))
#samples = np.tile(samples, days)

# Plot GP mean, confidence interval and samples 
plot_gp(mu_pr, cov, Xpr, samples=samples)
plt.xlim(0, 24)
plt.xticks(np.arange(0,26,2))
plt.xlabel("Hour")
#plt.ylim(0, 12)
plt.ylabel(r"EVs Connected")
plt.title(r"Monday Training Prior")
plt.tight_layout()

#%% --- Prediction from Noisey Data ---

# Noisy training data
days = 3;
dataFitTrain = dfFitTrain.head(days*24)
X = np.arange(0, 24, 1).reshape(-1, 1)
X_train = np.array(dataFitTrain.index.values).reshape(-1, 1)
X_train = X_train[0:days*24]
Y_train = np.array(dataFitTrain.Connected).reshape(-1, 1)
Y_train = Y_train[(days-1)*24:(days)*24]

# Compute mean and covariance of the posterior predictive distribution
mu_s, cov_s, sigY = posterior_predictive(X, X_train, Y_train)#, sigma_y=noise)

samples_s = np.random.poisson(mu.ravel(), size = (4,24))+1
#samples_s = np.tile(samples_s, day)

plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples_s)
plt.xlim(0, 24)
plt.xticks(np.arange(0,26,2))
plt.xlabel("Hour")
#plt.ylim(0, 12)
plt.ylabel(r"EVs Connected")
plt.title(r"Monday Posterior Prediction")
plt.tight_layout()

#%% Krasserm-Chan-BPR

l, sigma_f = 1.0, 1.0;

#X_train = np.arange(0,24,1).reshape(-1, 1)
#Y_train = np.array(dataFitTrain.Connected).reshape(-1, 1)
X_train = np.array(dfCount.Hour).reshape(-1, 1)
Y_train = np.array(dfCount.Connected).reshape(-1, 1)
#Y_train = Y_train[(days-1)*24:(days)*24]

Y_out = np.zeros((len(Xpr),1))
i = 0;
for h in Xpr.reshape(1,-1)[0]:
    #temp = dataFitTrain.loc[dataFitTrain.Hour == h]
    temp = dfCount.loc[dfCount.Hour == h]
    if len(temp) == 0:
        print('No Connects in Hr: ', h)
        Y_out[i] = 0;
    else:
        Y_out[i] = np.mean(temp.Connected)
    i += 1;

yNoise = 1/Y_out
yNoise[yNoise == np.inf] = 0
sigY = np.eye(len(yNoise))*(yNoise)
#sigY[sigY == -0.0] = 0.0    

#X_s = X
#X_train = np.arange(0, 24, 2).reshape(-1, 1)

K = kernel(Xpr, Xpr, l, sigma_f)# + sigY**2 * np.eye(len(X_train))
#K[K < 1e-8] = 0;
K_s = kernel(Xpr, X_train, l, sigma_f) #K_s is all zeros?
#K_s[K_s < 1e-8] = 0;
K_ss = kernel(X_train, X_train, l, sigma_f) + 1e-8 * np.eye(len(X_train))
#K_ss[K_ss < 1e-8] = 0;

# Equation (Kr4) to (Chan47)
K_sigY = K + sigY
K_sigY[K_sigY < 1e-8] = 0
#K_sigY = csc_matrix(K_sigY)

t = np.log(Y_out);
t[t < 0] = 0;

# Equation (Kr4)
mu_s = K_s.T.dot(inv(K_sigY)).dot(t)
#mu_s[mu_s < 0] = 0;

# Equation (Kr5) to (Chan48)
cov_s = K_ss - K_s.T.dot(inv(K + sigY)).dot(K_s)

#samples_s = np.random.poisson(mu_s.ravel(), size=(4,24))
samples_s = np.random.poisson(mu_s.ravel(), size=(4,len(mu_s)))
#samples_s = np.tile(samples_s, day)

#%% Plot_GP

X = dfCount.Hour

plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples_s)
plt.xlim(0, 24)
plt.xticks(np.arange(0,24,2))
plt.xlabel("Hour")
#plt.ylim(0, 12)
plt.ylabel(r"EVs Connected")
plt.title(r"Monday Posterior Prediction")
plt.tight_layout()


#%% Plot_GP 2

X = dfCount.Hour

dfPlot = pd.DataFrame()
dfPlot['Hour'] = dfCount.Hour
dfPlot['X_train'] = X_train; dfPlot['Y_train'] = Y_train;
dfPlot['mu'] = mu_s; dfPlot['cov'] = np.diag(cov_s);
dfPlot['Sample1'] = np.array(samples_s[0,:]).reshape(-1, 1)
dfPlot['Sample2'] = np.array(samples_s[1,:]).reshape(-1, 1)
dfPlot['Sample3'] = np.array(samples_s[2,:]).reshape(-1, 1)
dfPlot['Sample4'] = np.array(samples_s[3,:]).reshape(-1, 1)
dfPlot = dfPlot.sort_values(by=['Hour'])

sample_array = np.array(dfPlot.iloc[:,5:9]).reshape(4,-1);

plot_gp(dfPlot['mu'], dfPlot['cov'], dfPlot['X_train'], dfPlot['X_train'], dfPlot['Y_train'], samples=sample_array)
plt.xlim(0, 24)
plt.xticks(np.arange(0,24,2))
plt.xlabel("Hour")
#plt.ylim(0, 12)
plt.ylabel(r"EVs Connected")
plt.title(r"Monday Posterior Prediction")
plt.tight_layout()

#%% Plot

import seaborn as sns
sns.set(style="darkgrid")

dfPlot = pd.DataFrame()
dfPlot['Hour'] = dfCount.Hour
dfPlot['X_train'] = X_train; dfPlot['Y_train'] = Y_train;
dfPlot['mu'] = mu_s; dfPlot['cov'] = np.diag(cov_s);
dfPlot['Sample1'] = np.array(samples_s[0,:]).reshape(-1, 1)
dfPlot['Sample2'] = np.array(samples_s[1,:]).reshape(-1, 1)
dfPlot['Sample3'] = np.array(samples_s[2,:]).reshape(-1, 1)
dfPlot['Sample4'] = np.array(samples_s[3,:]).reshape(-1, 1)

dfPlot = dfPlot.sort_values(by=['Hour'])

# Plot the responses for different events and regions
ax = sns.lineplot(x="Hour", y="Y_train", data=dfPlot)

ax.set(xticks=np.arange(0,26,2), xlabel='Hour', ylabel='Connected')

ax.set_title('Monday Data')
