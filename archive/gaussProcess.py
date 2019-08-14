# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:20:19 2019

@author: Alex Palomino
https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py
"""

print(__doc__)

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, 
                                              RationalQuadratic,
                                              ExpSineSquared, 
                                              ConstantKernel, DotProduct, 
                                              Matern)


kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0)),
           ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
           1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)]

#krnl_ESS = 1.0 * ExpSineSquared(length_scale=1, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0))
#kernel = krnl_ESS;

kernels = [1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0))]

results = pd.DataFrame(index=np.arange(0,len(kernels)), columns=['Kernel','Params','LogLike'])
idx = 0;

for kernel in kernels:
        
    # Specify Gaussian Process
    # Note that the kernel’s hyperparameters are optimized during fitting.
    # Depending on the initial value for the hyperparameters, the gradient-based 
    # optimization might also converge to the high-noise solution. It is thus 
    # important to repeat the optimization several times for different initializations.
    gp = GaussianProcessRegressor(kernel=kernel)

    # Plot prior
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    X_ = np.linspace(0, 5, 100)
    y_mean, y_std, y_cov = gp.predict(X_[:, np.newaxis], return_std=False, return_cov=True)
    plt.plot(X_, y_mean, 'k', lw=3, zorder=1)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')
    y_samples = gp.sample_y(X_[:, np.newaxis], 10)
    plt.plot(X_, y_samples, lw=1)
    plt.xlim(0, 5)
    plt.ylim(-3, 3)
    plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

    # Generate data and fit GP
    rng = np.random.RandomState(4)
    X = rng.uniform(0, 5, 10)[:, np.newaxis]
    y = np.sin((X[:, 0] - 2.5) ** 2)
    gp.fit(X, y)

    # Plot posterior
    #plt.subplot(2, 1, 2)
    X_ = np.linspace(0, 5, 100)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    #plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    #plt.fill_between(X_, y_mean - y_std, y_mean + y_std, 
    #                 alpha=0.2, color='k')

    y_samples = gp.sample_y(X_[:, np.newaxis], 10)
    #plt.plot(X_, y_samples, lw=1)
    #plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    #plt.xlim(0, 5)
    #plt.ylim(-3, 3)
    #plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
    #          % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),
    #          fontsize=12)
    #plt.tight_layout()

    # Print Log-Likehood value for each kernel
    results.Kernel.at[idx] = str(kernel)
    results.Params.at[idx] = gp.kernel_.theta
    results.LogLike.at[idx] = gp.log_marginal_likelihood(gp.kernel_.theta)
    idx += 1;
    
plt.show()

#%% Original Reference Code

print(__doc__)

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
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

kernels = [1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0))]

for kernel in kernels:
    # Specify Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel)

    # Plot prior
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    X_ = np.linspace(0, 5, 100)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')
    y_samples = gp.sample_y(X_[:, np.newaxis], 10)
    plt.plot(X_, y_samples, lw=1)
    plt.xlim(0, 5)
    plt.ylim(-3, 3)
    plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

    # Generate data and fit GP
    rng = np.random.RandomState(4)
    X = rng.uniform(0, 5, 10)[:, np.newaxis]
    y = np.sin((X[:, 0] - 2.5) ** 2)
    gp.fit(X, y)

    # Plot posterior
    plt.subplot(2, 1, 2)
    X_ = np.linspace(0, 5, 100)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')

    y_samples = gp.sample_y(X_[:, np.newaxis], 10)
    plt.plot(X_, y_samples, lw=1)
    plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    plt.xlim(0, 5)
    plt.ylim(-3, 3)
    plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
              % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),
              fontsize=12)
    plt.tight_layout()

plt.show()

result0 = [str(kernel), gp.kernel_.theta, gp.log_marginal_likelihood(gp.kernel_.theta)]

print('Default Result \n  ', result0)

#%%

import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(0.1, 5.0))

gp = GaussianProcessRegressor(kernel=kernel)

# Plot prior
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
X_ = np.linspace(0, 5, 100)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                 alpha=0.2, color='k')
y_samples = gp.sample_y(X_[:, np.newaxis], 10)
plt.plot(X_, y_samples, lw=1)
plt.xlim(0, 5)
plt.ylim(-3, 3)
plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

# Generate data and fit GP
X = np.arange(0,24)[:, np.newaxis]
y = data[2]
gp.fit(X, y)

# Plot posterior
plt.subplot(2, 1, 2)
X_ = np.linspace(0, 5, 100)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                 alpha=0.2, color='k')

y_samples = gp.sample_y(X_[:, np.newaxis], 10)
plt.plot(X_, y_samples, lw=1)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
plt.xlim(0, 5)
plt.ylim(-3, 3)
plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
          % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),
          fontsize=12)
plt.tight_layout()

#%%
"""
Created on Wed Jun 12 11:25:19 2019

@author: Alex Palomino
https://app.dominodatalab.com/u/fonnesbeck/gp_showdown/view/GP+Showdown.ipynb

http://localhost:8888/notebooks/Documents/GitHub/EV-Flexibility/jupyter/GP_Showdown.ipynb
"""

#matplotlib inline
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pylab as plt

np.random.seed(42)

def exponential_cov(x, y, params):
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)

def conditional(x_new, x, y, params):
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)
    A = exponential_cov(x_new, x_new, params)
    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
    return(mu.squeeze(), sigma.squeeze())
    
θ = [1, 10]
σ_0 = exponential_cov(0, 0, θ)
xpts = np.arange(-3, 3, step=0.01)
plt.errorbar(xpts, np.zeros(len(xpts)), yerr=σ_0, capsize=0)
plt.ylim(-3, 3);

#%%

x = [1.]
y = [np.random.normal(scale=σ_0)]
print(y)

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
plt.xlim(-3, 3); plt.ylim(-3, 3);   

#%%

m, s = conditional([-0.7], x, y, θ)
y2 = np.random.normal(m, s)
y2

x.append(-0.7)
y.append(y2)

σ_2 = exponential_cov(x, x, θ)

predictions = [predict(i, x, exponential_cov, θ, σ_2, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")
plt.xlim(-3, 3); plt.ylim(-3, 3);

#%%

x_more = [-2.1, -1.5, 0.3, 1.8, 2.5]
mu, s = conditional(x_more, x, y, θ)
y_more = np.random.multivariate_normal(mu, s)
y_more

x += x_more
y += y_more.tolist()

σ_new = exponential_cov(x, x, θ)

predictions = [predict(i, x, exponential_cov, θ, σ_new, y) for i in x_pred]

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")
plt.ylim(-3, 3);

#%%
# https://plot.ly/scikit-learn/plot-gpr-noisy-targets/

import sklearn
import plotly 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import *

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#np.random.seed(1)

def f(x):
    """The function to predict."""
    return x * np.sin(x)

def data_to_plotly(x):
    k = []
    
    for i in range(0, len(x)):
        k.append(x[i][0])
        
    return k

zX = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
zy = f(zX).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
zx = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instanciate a Gaussian Process model
zkernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
zgp = GaussianProcessRegressor(kernel=zkernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
zfit = zgp.fit(zX, zy)
print(zfit)

# Make the prediction on the meshed x-axis (ask for MSE as well)
zy_pred, zsigma = zgp.predict(zx, return_std=True)

#%% Plot

zp1 = go.Scatter(x=data_to_plotly(zx), y=data_to_plotly(f(zx)), 
                mode='lines',
                line=dict(color='red', dash='dot'),
                name=u'<i>f(x) = xsin(x)</i>')

zp2 = go.Scatter(x=data_to_plotly(zX), y=zy, 
               mode='markers',
               marker=dict(color='red'),
               name=u'Observations')

zp3 = go.Scatter(x=data_to_plotly(zx), y=zy_pred, 
                mode='lines',
                line=dict(color='blue'),
                name=u'Prediction',
               )

zp4 = go.Scatter(x=data_to_plotly(np.concatenate([zx, zx[::-1]])),
                y=np.concatenate([zy_pred - 1.9600 * zsigma,]),
                mode='lines',
                line=dict(color='blue'),
                fill='tonexty',
                name='95% confidence interval')


zdata = [zp3, zp4, zp1, zp2]
zlayout = go.Layout(xaxis=dict(title='<i>x</i>'),
                   yaxis=dict(title='<i>f(x)</i>'),
                  )
zfig = go.Figure(data=zdata, layout=zlayout)

#%%
#py.iplot(fig)
plotly.offline.plot(zfig)