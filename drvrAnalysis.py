# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:18:43 2020

@author: Alex Palomino
"""

import pandas as pd
import numpy as np
from itertools import combinations 

chgrDistances = pd.read_excel('data/charger_distances.xlsx');

#%% Basic Driver Metrics

#dfTemp = dict_seshDrvr['3820891']
#plt.scatter(dfTemp['Start Date'], dfTemp['Energy (kWh)'])# s=dfTemp['Energy (kWh)'])

df_metrics = pd.DataFrame(np.empty((len(drvr_include),8)), 
                          index=drvr_include, columns=['DrvrZip','AvgEnergy','AvgDur','AvgChrg','Ratio','AvgStart','NumChgrs','MaxDist'])
i = 0;
for d in drvr_include:
    df_metrics.loc[d].DrvrZip = dict_seshDrvr[d].iloc[0]['Driver Postal Code'];
    df_metrics.AvgEnergy.at[d] = dict_seshDrvr[d]['Energy (kWh)'].mean();
    df_metrics.AvgDur.at[d] = dict_seshDrvr[d]['Duration (h)'].mean();
    df_metrics.AvgChrg.at[d] = dict_seshDrvr[d]['Charging (h)'].mean();
    df_metrics.Ratio.at[d] = df_metrics.AvgChrg.at[d]/df_metrics.AvgDur.at[d];
    df_metrics.AvgStart.at[d] = dict_seshDrvr[d]['StartHr'].mean();
    df_metrics.NumChgrs.at[d] = len(list(set(dict_seshDrvr[d]['EVSE ID']))); 
    i+=1;

print('--- Summary ---\n',df_metrics.describe())

#%% Calculate Maximum Distance between Chargers Used by a Single Driver

dict_maxDist = {};
for d in drvr_include:
    
    EVSE_pairs = list(combinations(list(set(dict_seshDrvr[d]['EVSE ID'])),2))
    
    dist = {}; maxDist = 0;
    for p in EVSE_pairs:

        if len(chgrDistances.loc[chgrDistances['EVSE ID'] == str(p)]) == 1:
            dist[p] = float(chgrDistances.loc[chgrDistances['EVSE ID'] == str(p)].Dist.values[0].split()[0]);
            
        else:
            dist[p] = float(chgrDistances.loc[chgrDistances['EVSE ID'] == str(p[::-1])].Dist.values[0].split()[0]);
    
    if len(dist) > 0:
        maxDist = max(dist.values());
        df_metrics.MaxDist.at[d] = maxDist;
        
    else:
        maxDist = ' not found.';
        df_metrics.MaxDist.at[d] = 0;
        
    dict_maxDist[d] = maxDist;
                
    print('Driver :', d, ' | ', dict_maxDist[d])
    
print('--- Summary ---\n',df_metrics.describe())
summary = df_metrics.describe()

df_metrics.to_excel("data/driver_metrics.xlsx")      
#%% Start Clustering

# import libraries
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from scipy.spatial import distance
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import timeit

start_time = timeit.default_timer()

clusters = 4
colNames = ['AvgEnergy','AvgDur','AvgChrg','Ratio','AvgStart','NumChgrs','MaxDist']
df = df_metrics.filter(colNames, axis=1)

# Get column names first
names = df.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=names)

# Calculate k-Means
kmeans = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(df_scaled)

#phi_true = kmeans.labels_
phi_predict = kmeans.predict(df_scaled)

centers = kmeans.cluster_centers_
score = kmeans.score(df_scaled)

# Compute Clustering Metrics
n_clusters_ = len(centers)

print('Number of clusters: %d' % n_clusters_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(phi_true, phi_predict))
#print("Completeness: %0.3f" % metrics.completeness_score(phi_true, phi_predict))
#print("V-measure: %0.3f" % metrics.v_measure_score(phi_true, phi_predict))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(phi_true, phi_predict))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(phi_true, phi_predict))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df_scaled, phi_predict, metric='sqeuclidean'))

# timeit statement
elapsed = timeit.default_timer() - start_time

#%%

import math

df = df_metrics;
plt.scatter(df.AvgStart,df.AvgEnergy)

#%%
def round10(x):
    return int(math.ceil(x / 5.0)) * 5

df.AvgEnergy = df.AvgEnergy.apply(lambda x: round10(x))