#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION¶
# ## TASK 2 - Prediction using Unsupervised ML
# Predicting the optimum number of clusters and representing it visually using K-means clustering.
# 
# ### Name : Emna Fathalli

# In[1]:


#importing libraries
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


# In[2]:


dataset = pd.read_csv("Iris.csv", index_col=0)
dataset.head(5)


# In[3]:


set(dataset.Species.values)


# In[4]:


dataset.describe()


# In[5]:


#checking for null values
dataset.isnull().sum()


# In[6]:


#dropping ALL duplicte values
dataset.drop_duplicates(inplace=True)


# ### Data Visualisation

# In[7]:


sns.pairplot(dataset,hue ='Species')


# In[8]:


#select a particular cell of the dataset using iloc function
X = dataset.iloc[:, [0,1,2, 3]].values


# In[9]:


# Using the elbow method to find the optimal number of clusters
wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[10]:


plt.plot(range(1, 11), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# To determine the optimal number of clusters, the value of k at the elbow is selected i.e. the point after which the distortion start decreasing in a linear fashion. Thus form the above graph, it can be said that the optimum clusters the data is 3.

# In[11]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# In[12]:


#visualising the data
#plt.figure(figsize=(9,8))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'red', label = 'Iris-setosa',s=120)
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Iris-versicolour',s=120)
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Iris-virginica',s=120)

## Plotting the centroids of the clusters
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'yellow', label = 'Centroids',s=100)
plt.grid(False)
plt.title('Clusters of Iris')
plt.legend()
plt.show()


# In[ ]:




