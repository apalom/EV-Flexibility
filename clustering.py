# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:36:51 2019

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
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn import metrics

#%% Import Data

# Raw Data
filePath = 'exports/EVSE_Heuristics-All50k_pop.xlsx';

# Import Data
data = pd.read_excel(filePath, index_col='EVSE ID');

#dataHead = data.head(100);
#dataTypes = data.dtypes;

allColumns = list(data);

#%% Normalize Data

def normalize(df, colNames):
    
    df = df.filter(colNames, axis = 1)
    df = df.loc[df['Energy (kWh)'] > 0]
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=colNames)
    
    return df;

colNames = ['Energy (kWh)', 'Duration (h)', 'Charging (h)', 
            'DayofWk', 'StartHr', 'EndHr', 'AvgPwr',]

df_N = normalize(dfAll,colNames)

#%% Cluster

def cluster(df_N, max_cluster):
    
    start_time = timeit.default_timer()
    
    sil_coef = {}
    
    for k in np.arange(2,max_cluster,1):
    
        # Calculate k-Means
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300).fit(df_N)
        
        phi_true = kmeans.labels_
        phi_predict = kmeans.predict(df_N)
        
        centers = kmeans.cluster_centers_
        score = kmeans.score(df_N)
        
        # Compute Clustering Metrics
        n_clusters_ = len(centers)
            
        sil_coef[k] = metrics.silhouette_score(df_N, phi_predict, metric='sqeuclidean')
       
        
        print("Silhouette Coefficient: ", sil_coef)
    
     # timeit statement
    elapsed = timeit.default_timer() - start_time
    print('Execution time: {0:.4f} sec'.format(elapsed))
        
    x, y = zip(*sorted(sil_coef.items())) # unpack a list of pairs into two tuples
    
    plt.plot(x, y)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    
    plt.show()
    
    return sil_coef;

sil_coef = cluster(df_N, 10)

#%%

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(phi_true, phi_predict))
print("Completeness: %0.3f" % metrics.completeness_score(phi_true, phi_predict))
print("V-measure: %0.3f" % metrics.v_measure_score(phi_true, phi_predict))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(phi_true, phi_predict))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(phi_true, phi_predict))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(dfHeurN, phi_predict, metric='sqeuclidean'))