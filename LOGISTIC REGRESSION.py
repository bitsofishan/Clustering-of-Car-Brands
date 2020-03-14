# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:02:31 2020

@author: Ishan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cars=pd.read_csv(r'C:\Users\Ishan\Documents\Python Scripts\Datasets\Cars(K-means Clustering).csv')
print(cars)

#X=cars.iloc[:,:-1].values
X=cars[cars.columns[:-1]]
#X=pd.to_numeric(X,errors='ignore')
X=X.apply(pd.to_numeric,errors='coerce')
print(X.head)

for i in X.columns:
    X[i]=X[i].fillna(int(X[i].mean()))
for i in X.columns:
    print(X[i].isnull().sum())
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow Method")
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")
plt.show()
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=kmeans.fit_predict(X)
X=X.as_matrix(columns=None)
print(y_means)
plt.scatter(X[y_means==0,0],X[y_means==0,1],s=100,c='red',label="Japan")
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100,c='blue',label="US")
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=100,c='green',label="Europe")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow",label="centroids")
plt.legend()
plt.show()






