# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:56:40 2020

@author: Alex Palomino
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
"""

#%% Load MNIST Digit Image Data (70,000 qty 784 x 784 pixel images)
# mnist.data and has a shape of (70000, 784). There are 70,000 images 
# with 784 dimensions (784 features).
# The labels (the integers 0–9) are contained in mnist.target. 
# The features are 784 dimensional (28 x 28 images) and the labels 
# are simply numbers from 0–9.

import sklearn
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)

#%% Train/Test Split (6/7 and 1/7)

from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

#%% Standardize the Data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#%% Principal Component Analysis

from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.95)

# Fit PCA on Training Data
pca.fit(train_img)

# Apply PCA Mapping
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

#%% Apply Logistic Regression to the Transformed Data

from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(max_iter=1000)

# Training the model on the data, storing the information learned from the data
logisticRegr.fit(train_img, train_lbl)

# Predict for Multiple Observations (images)
test_pred = logisticRegr.predict(test_img[0:10])

#%% Measure Model Performance

score = logisticRegr.score(test_img, test_lbl)
