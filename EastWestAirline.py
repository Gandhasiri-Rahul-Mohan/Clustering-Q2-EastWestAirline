# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:32:21 2022

@author: Rahul
"""

import pandas as pd
import numpy as np

df = pd.read_excel("D:\\DS\\books\\ASSIGNMENTS\\Clustering\\EastWestAirlines.xlsx", sheet_name = 'data')
df

df = df.drop(['ID#'],axis=1)
df
df.shape
df.dtypes
df.info()
df.isnull().sum()
# There is no null values

# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
# Normalized data frame (considering the numerical part of data)
X = norm_func(df.iloc[:,0:])
X.describe()

# Hierarchical clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # for creating dendrogram 
z = linkage(X, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('EWAirlines')
sch.dendrogram(z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,)  # font size for the x axis labels
plt.show()
# The dendrogram shows 3 main clusterings with some outliers
# Since the data is huge we cannot decide, but lets chose the no. of clusters as 5

from sklearn.cluster import AgglomerativeClustering
AC=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
Y=AC.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()
df2=pd.concat([X,Y],axis=1)

# K-Means clustering 
# To check how many clusters are requried
from sklearn.cluster import KMeans
inertia = []
for i in range(1, 10):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)
print(inertia)

# Elbow method to see variance in inertia by clusters
plt.plot(range(1, 10), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()
# From the graph we can see that the optimal number of clusters is 5

# Scree plot
import seaborn as sns
d1 = {"kvalue": range(1, 10),'inertiavalues':inertia}
d2 = pd.DataFrame(d1)
sns.barplot(x='kvalue',y="inertiavalues", data=d2) # kvalue=clusters
# Here the variance in inertia b/w 5th and 6th cluster is less so we can go with 5 clusters

KM=KMeans(n_clusters=5,n_init=10,max_iter=300)
Y=KM.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()
df1=pd.concat([X,Y],axis=1)

# DBSCAN Clustering ###
# Normalize heterogenous numerical data using standard scalar fit transform to dataset
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
df_norm=StandardScaler().fit_transform(df)
df_norm
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(df_norm)
# Noisy samples are given the label -1.
dbscan.labels_
# Adding clusters to dataset
df['clusters']=dbscan.labels_
df
df.groupby('clusters').agg(['mean']).reset_index()
# The output are outliers/noise data after removing this we can apply any other 
# clustering techniques for better output

# In Hierarchical clustering, dendrograms are a drawback if the data is huge.
# Here, since data is huge, hierarchical clustering is not considered as a good option. 
# K-means is considered as the simplest and quickest one which guarantees convergence.

































