# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:58:48 2019

@author: Alex

http://krasserm.github.io/2018/03/19/gaussian-processes/
Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning.
Bayesian Poisson Regression for Crowd Counting [Chan]
"""

import numpy as np

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    ''' Isotropic squared exponential kernel (RBF). Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


from numpy.linalg import inv

def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    ''' Computes the suffifient statistics of the GP posterior predictive distribution from m training data X_train and Y_train and n new inputs X_s. Args: X_s: New input locations (n x d). X_train: Training locations (m x d). Y_train: Training targets (m x 1). l: Kernel length parameter. sigma_f: Kernel vertical variation parameter. sigma_y: Noise parameter. Returns: Posterior mean vector (n x d) and covariance matrix (n x n). '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    sigY = inv(np.eye(len(Y_train)).dot(Y_train))
    #K_inv = inv(K)
    
    # Equation (Kr4) to (Chan47)
    mu_s = K_s.T.dot(inv(K + sigY)).dot(np.log(Y_train))

    # Equation (Kr5) to (Chan49=8)
    cov_s = K_ss - K_s.T.dot(inv(K + sigY)).dot(K_s)
    
    return mu_s, cov_s