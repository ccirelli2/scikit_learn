''' Kmeans tutorial
    URL:  https://towardsdatascience.com/customer-segmentation-with-machine-learning-a0ac8c3d4d84
    URL:  https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/

    Unsupervised:
        We do not have a target to predict.  We look at the data and then try to group similar
        observations and form different groups.  

    Clustering Applications:
        - Customer Segmentation: Banking, recommendation engines, document clusters, 
            image segmentation. 
        - Documentation Clustering:  This could be an easy practice project using
            books or documents by type.
        - Image Segmentation: Group similar pixels in the same group.  
        - Recommendation Engines:  Use songs liked by people and then use clustering
            to find similar songs and then make a recommendation by those that are
            most similar. 

    Cluster Evaluation Metrics:

    1.) Inertia:    It tells us how far apart the points within a cluster are (so spread?)
                    Calculates the sum of distances of all points within a cluster from the 
                    centroid of that cluster. 
                    We calculate this for all of the clusters and then final inertia value
                    is the sum of all of these distances. 
                    *Distance within a cluster is known as "intra-cluster distance"
                    *We can say that the less the inertia value, the better the cluster. 
    2.) Dunn Index: 
                    Deals with the principal that different clusters should be as
                    different from one another as possible. 
                    Dunn Index = min(Inter Clusters Distance) /
                                 max(Intra Cluster Distance)

    Questions:
        What is the purpose of clustering if you don't have a target?  For instance, you
        want to know which are the most profitable customers?  Is Kmeans not appropriate
        as you have a target variable?

'''

# IMPORT LIBRARIES -------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# LOAD DATA --------------------------------------------------
dir_data = r'/home/ubuntu/Desktop/repositories/scikit_learn/data'
afile = r'customer_data.csv'
df = pd.read_csv(dir_data + '/' + afile)
pd.set_option('display.max_columns', None)


# EXPLORATORY DATA ANALYSIS ----------------------------------

def eda(df):
    print(df.head(), '\n')
    print(df.describe(), '\n')
    print('Count of Null Values: \n', df.isnull().sum())


# PREPROCESSING (SCALE) DATA ---------------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(data_scaled)


# INSANTIATE CLUSTER -----------------------------------------

kmeans = KMeans(n_clusters=2, init='k-means++')
kmeans.fit(data_scaled)


# EVALUATION -------------------------------------------------
print(kmeans.inertia_)


# Optomize

def optomize_model(data_scaled):
    sse = []

    for cluster in range(1,20):
        kmeans = KMeans(n_clusters = cluster, init='k-means++')
        kmeans.fit(data_scaled)
        sse.append(kmeans.inertia_)

    plt.plot(sse)
    plt.title('Kmeans Inertia by Number of Clusters')
    plt.show()

optomize_model(data_scaled)





