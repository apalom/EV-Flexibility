# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:28:45 2019

@author: Alex
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
data = pd.read_excel('exports\DataForAvi.xlsx', sheet_name='Wkdy-Hr1');
data = data.drop('Hr', axis=1);

#%% Make GPR Prediction

import sklearn
import plotly 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import *

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Discrete Space
X = np.atleast_2d(np.arange(0,24)).T

# Observations
y = data.iloc[:,np.random.randint(0,data.shape[1])]

# Continuous Space
x = np.atleast_2d(np.linspace(0, 23, 2300)).T

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

# Fit to data using Maximum Likelihood Estimation of the parameters
fit = gp.fit(X, y)
print(fit)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

#%% Setup  Plot

p1 = go.Scatter(x=data_to_plotly(x), y=data_to_plotly(f(x)), 
                mode='lines',
                line=dict(color='red', dash='dot'),
                name=u'<i>Observation Fit</i>')

p2 = go.Scatter(x=data_to_plotly(X), y=y, 
               mode='markers',
               marker=dict(color='red'),
               name=u'Observations')

p3 = go.Scatter(x=data_to_plotly(x), y=y_pred, 
                mode='lines',
                line=dict(color='black'),
                name=u'Prediction',
               )

p4 = go.Scatter(x=data_to_plotly(np.concatenate([x, x[::-1]])),
                y=np.concatenate([y_pred - 1.9600 * sigma,]),
                mode='lines',
                line=dict(color='blue'),
                fill='tonexty',
                name='95% confidence interval')


data = [p3, p4, p1, p2]
layout = go.Layout(xaxis=dict(title='<i>x</i>'),
                   yaxis=dict(title='<i>f(x)</i>'),
                  )
fig = go.Figure(data=data, layout=layout)

# Send to Plotly

plotly.offline.plot(fig)

#%%

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause

import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0)),
           ConstantKernel(0.1, (0.01, 10.0))
               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
           1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)]

for kernel in kernels:
    # Specify Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel)

    # Plot prior
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    
    Xc = np.linspace(0,23,230) 
    Xd = np.linspace(0,23,24)
    y_mean = np.mean(data, axis=1)
    y_std = np.std(data, axis=1)

    plt.plot(Xd, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(Xd, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')    
    
    # Negative values are clipped    
    y_samples = gp.sample_y(data, 10).clip(min=0)
    
    # Plot Prior
    plt.plot(Xd, y_samples, lw=1)
    plt.xticks(np.arange(0,24,2))
    plt.xlim(0, 24)
    plt.ylim(0, 20)
    plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

    # Generate data and fit GP    
    X = np.arange(0,24)[:, np.newaxis]
    y = data[np.random.randint(0,data.shape[1])][:, np.newaxis]
    gp.fit(X, y)

    # Plot posterior
    plt.subplot(2, 1, 2)
    y_meanP, y_stdP = gp.predict(Xc[:, np.newaxis], return_std=True)
    plt.plot(Xc, y_meanP, 'k', lw=3, zorder=9)
    plt.fill_between(Xc, y_meanP - y_stdP, y_meanP + y_stdP,
                     alpha=0.2, color='k')

    y_samplesP = gp.sample_y(Xc, 10)
    plt.plot(Xc, y_samplesP, lw=1)
    plt.scatter(Xd[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    plt.xlim(0, 24)
    plt.ylim(0, 20)
    plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
              % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),
              fontsize=12)
    plt.tight_layout()

plt.show()

result0 = [str(kernel), gp.kernel_.theta, gp.log_marginal_likelihood(gp.kernel_.theta)]
print('Default Result \n  ', result0)


#%% 
# http://krasserm.github.io/2018/03/19/gaussian-processes/

import numpy as np

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

# Finite number of points
X = np.arange(-1, 1, 0.2).reshape(-1, 1)

# Mean and covariance of the prior
mu = np.zeros(X.shape)
cov = kernel(X, X)

# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

# Plot GP mean, confidence interval and samples 
plot_gp(mu, cov, X, samples=samples)

#%% https://blog.dominodatalab.com/fitting-gaussian-process-models-python/

import numpy as np
 
# All we will do here is sample from the prior Gaussian process, so before any data have been
# introduced. What we need first is our covariance function, which will be the squared exponential, 
# and a function to evaluate the covariance at given points (resulting in a covariance matrix)

def exponential_cov(x, y, params):
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)

# We are going generate realizations sequentially, point by point, using the lovely conditioning
# property of mutlivariate Gaussian distributions.
# And this the function that implements it.

def conditional(x_new, x, y, params):
 
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)
    A = exponential_cov(x_new, x_new, params)
     
    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
     
    return(mu.squeeze(), sigma.squeeze())

# We will start with a Gaussian process prior with hyperparameters $\theta_0=1, \theta_1=10$. 
# We will also assume a zero function as the mean, so we can plot a band that represents one 
# standard deviation from the mean.

import matplotlib.pylab as plt
 
θ = [1, 10]
σ_0 = exponential_cov(0, 0, θ)
xpts = np.arange(-3, 3, step=0.01)
plt.errorbar(xpts, np.zeros(len(xpts)), yerr=σ_0, capsize=0, alpha=0.25)
plt.ylim(-2,2)

# Let’s select an arbitrary starting point to sample, say $x=1$. Since there are no prevous 
# points, we can sample from an unconditional Gaussian.
x = [1.]
y = [np.random.normal(scale=σ_0)]
print('y = ', y)

# We can now update our confidence band, given the point that we just sampled, using the covariance 
# function to generate new point-wise intervals, conditional on the value $[x_0, y_0]$.

σ_1 = exponential_cov(x, x, θ)
 
def predict(x, data, kernel, params, sigma, t):
    
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)
    y_pred = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
    
    return y_pred, sigma_new
 
x_pred = np.linspace(-3, 3, 1000)
predictions = [predict(i, x, exponential_cov, θ, σ_1, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")

# So conditional on this point, and the covariance structure we have specified, we have 
# essentially constrained the probable location of additional points. Let’s now sample another.

m, s = conditional([-0.7], x, y, θ)
y2 = np.random.normal(m, s)
print('y2 = ', y2)

# This point is added to the realization, and can be used to further update the 
# location of the next point.

x.append(-0.7)
y.append(y2)
 
σ_2 = exponential_cov(x, x, θ)
predictions = [predict(i, x, exponential_cov, θ, σ_2, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")

# Of course, sampling sequentially is just a heuristic to demonstrate how the covariance 
# structure works. We can just as easily sample several points at once

x_more = [-2.1, -1.5, 0.3, 1.8, 2.5]
mu, s = conditional(x_more, x, y, θ)
y_more = np.random.multivariate_normal(mu, s)
print('y_more', y_more)

x += x_more
y += y_more.tolist()
 
σ_new = exponential_cov(x, x, θ)
predictions = [predict(i, x, exponential_cov, θ, σ_new, y) for i in x_pred]
 
y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")

#%% scikit-learn | https://blog.dominodatalab.com/fitting-gaussian-process-models-python/ 

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

num = 51
x = np.array(10 * np.random.random((1,51)) - 5)
X = x.reshape(-1, 1)
X.shape

y = 2*x + 2*np.random.random()*np.sin(2*np.pi*x) 
Y = y.reshape(-1, 1)

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp.fit(X, Y)

gp.kernel_

x_pred = np.linspace(-6, 6).reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

plt.scatter(X,Y)
plt.plot(x_pred,y_pred)


#%% Working

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)