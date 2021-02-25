#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS Objective: Predict the win maker. 
"""

# Import Libraries 
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import logging
import inspect
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import xgboost as xgb

pd.set_option('display.max_columns', None)

# Load Data -----------------------------------------------
wine_load = datasets.load_wine()
out_dir = r'/home/ccirelli2/Desktop/repositories/Scikit_Learn/output'
 

# Basic Data Inspection -----------------------------------
def get_basic_dataset_info(data_load):
    print('Dataset Description \n', data_load['DESCR'])
    print('Wind dataset keys => {}\n'.format(data_load.keys()))
    print('Target names => {}\n'.format(data_load['target_names']))
    print('Target values => {}\n'.format(data_load['target']))


# Get Feature & Target Variables
X, y = datasets.load_wine(True)
feature_names = wine_load['feature_names']



# Exploratory Data Analysis -------------------------------------

# Create pandas dataframe of dataset
'''https://scikit-learn.org/stable/modules/preprocessing.html'''
def create_df_data(x,y,feature_names):
    df = pd.DataFrame({})
    for n in range(0, len(feature_names)):
        feature = feature_names[n]
        data    = x[:, n]
        df[feature] = data
    df['target'] = y
    return df

df_wine = create_df_data(X,y,feature_names)

# Check Differing Means of each feature w/r/t the targets
def compare_means_targets_features(df_wine, feature_names, pprint=True, plot=True):
    ''' Takeaway:  The key takeaway is that there are differences between the 
                    means of each feature and target variable.  So there is a good
                    chance that these features will have some prediction power. 
    '''
    group_target_mean_feature = df_wine.groupby(['target']).mean()
     
    if pprint == True:
        print(group_target_mean_feature.loc[0])
    if plot == True:
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set3')
        for col in range(0, len(df_wine.columns)-1): 
            plt.subplot(4,4, col+1)
            group_target_mean_feature.iloc[:, col].plot.bar()
            plt.title(feature_names[col])
        plt.show()


# Check the distribution of the features
def plot_hist_features(df_wine):
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set3')
    for col in range(0, len(df_wine.columns)-1):
        plt.subplot(4,4, col+1)
        df_wine.iloc[:, col].plot.hist()
        plt.title(feature_names[col])
    plt.show()


# Note: Always perform a chi-squared test on categorical variables.  see tuturial. 



# Split Train / Test --------------------------------------------------
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, 
        random_state = 47)

# Instantiate & Fit Model --------------------------------------------
rdf = RandomForestClassifier(n_estimators=1000, bootstrap=True)
rdf_fit = rdf.fit(X_train, Y_train)
rdf_pred = rdf_fit.predict(x_test)
print(accuracy_score(y_test, rdf_pred))

# Print Classification Report
class_report = metrics.classification_report(y_test, rdf_pred, labels = [0,1,2])
conf_matrix = metrics.confusion_matrix(y_test, rdf_pred, labels=[0,1,2])
print(class_report)
print(conf_matrix)


# Vizualize Tree -------------------------------------------------------
def viz_tree():
    atree = rdf_fit.estimators_[5]
    export_graphviz(atree, out_file= out_dir + '/' + 'rf_wine_plot_tree.dot', 
            feature_names = feature_names, rounded=True, precision=1)

viz_tree()







