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

# Posterior Prediction
def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    ''' Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s. 
    Args: X_s: New input locations (n x d). X_train: Training locations (m x d). 
    Y_train: Training targets (m x 1). l: Kernel length parameter. 
    sigma_f: Kernel vertical variation parameter. sigma_y: Noise parameter. 
    Returns: Posterior mean vector (n x d) and covariance matrix (n x n). '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    sigY = inv(np.eye(len(Y_train))*(Y_train))
    sigY[sigY == -0.0] = 0.0
    #K_inv = inv(K)
    
    # Equation (Kr4) to (Chan47)
    mu_s = K_s.T.dot(inv(K + sigY)).dot(np.log(Y_train))

    # Equation (Kr5) to (Chan48)
    cov_s = K_ss - K_s.T.dot(inv(K + sigY)).dot(K_s)
    
    return mu_s, cov_s, sigY

#%% Krasserm Bayesian ML GP Utility
# https://github.com/krasserm/bayesian-machine-learning/blob/af6882305d9d65dbbf60fd29b117697ef250d4aa/gaussian_processes_util.py#L7
    
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)

#%% --- Prior ---
    
# Finite number of points
X = np.arange(0, 10, 0.2).reshape(-1, 1)

# Mean and covariance of the prior
mu = 2 + np.zeros(X.shape)
cov = kernel(X, X)

# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

# Plot GP mean, confidence interval and samples 
plot_gp(mu, cov, X, samples=samples)
plt.tight_layout()

#%% --- Prediction from Noisey Data ---

noise = 0.0

# Noisy training data
X_train = np.arange(-0, 10, 1).reshape(-1, 1)
#Y_train= np.arange(1,6).reshape(-1, 1)
Y_train = 2 + np.sin(X_train) + noise * np.random.randn(*X_train.shape)

# Compute mean and covariance of the posterior predictive distribution
mu_s, cov_s, sigY = posterior_predictive(X, X_train, Y_train, sigma_y=noise)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)













