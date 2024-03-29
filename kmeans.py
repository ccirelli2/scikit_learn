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
    1.) What is the purpose of clustering if you don't have a target?  For instance, you
        want to know which are the most profitable customers?  Is Kmeans not appropriate
        as you have a target variable?
    2.) How can be best understand what attributes gave risk the to the definition of the 
        clusters, i.e. what features are most driving the differentiation?
'''


# IMPORT LIBRARIES -------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# LOAD DATA --------------------------------------------------
dir_data = r'/home/cc2/Desktop/repositories/scikit_learn/data'
afile = r'customer_data.csv'
df = pd.read_csv(dir_data + '/' + afile)
pd.set_option('display.max_columns', None)
col_names = df.columns

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

kmeans = KMeans(n_clusters=5, init='k-means++')
kmeans.fit(data_scaled)


# EVALUATION -------------------------------------------------
print('Inertia => {}'.format(kmeans.inertia_))


# FIND OPTIAL NUMBER OF CLUSTERS -----------------------------
' Note:  Looking for an elbow point.  Occurs between 5-8'
def optomize_model(data_scaled):
    sse = []

    for cluster in range(1,20):
        kmeans = KMeans(n_clusters = cluster, init='k-means++')
        kmeans.fit(data_scaled)
        sse.append(kmeans.inertia_)

    plt.plot(sse)
    plt.title('Kmeans Inertia by Number of Clusters')
    plt.show()



# ASSIGN CLUSTER VALUES TO DATA ------------------------------

# Generate Prediction 
pred = kmeans.predict(data_scaled)

# Undo Scaling
df_unscaled = pd.DataFrame(scaler.inverse_transform(df_scaled))

# Add Prediction Values to Dataframe
df_unscaled['cluster_num'] = pred
df_unscaled.rename(columns={0:col_names[0], 1:col_names[1], 2:col_names[2], 3:col_names[3], 
    4:col_names[4], 5:col_names[5], 6:col_names[6], 7:col_names[7]}, inplace=True)

# Plot Cluster Number Distribution

def plot_groupby_mean(group1, group2):
    df_groupby_cluster_region = df_unscaled.groupby([group1, group2]).mean()
    df_groupby_cluster_region.plot(kind='bar')
    plt.title('Group by => {} and => {}'.format(group1, group2))
    plt.show()

plot_groupby_mean('Region', 'cluster_num')

