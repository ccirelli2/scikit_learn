#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:  Practice with exploratory data analysis techniques

Boston dataset
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
"""

# Import Libraries
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import scipy
import numpy as np
import math as m
import seaborn as sns
import pandas as pd

# How to Load Datasets
boston_load = datasets.load_boston
housing_load= datasets.california_housing
iris_load   = datasets.load_iris

# Get Description of Dataset
def print_desc_dataset(data):
    print(data)
    print(data.get("DESCR"))
    print(data.get('feature_names'))
    print(data.get('filename'))


# Get X & Y Data
boston_x, boston_y =     sklearn.datasets.load_boston(return_X_y=True)

# Create DataFrame
from sklearn.datasets import load_boston
boston = load_boston()
df_boston = pd.DataFrame(boston['data'], columns=boston['feature_names'])
df_boston['median_price'] = boston_y

print(df_boston.head())

# Investigate Shape of Data
print('Type of object', type(boston_x))
print('List of Attributes => {}'.format(dir(iris_load())))
print('Shape of X', boston_x.shape)
print('Shape of Y', boston_y.shape)

# Print Head of Features
def get_summary_statistics(np_array, data):
    num_features = len(np_array[0,:])
    for item in zip(range(0, num_features), data.get('feature_names')):
        data_desc = scipy.stats.describe(np_array[:, item[0]])
        print(item[1], '\n', data_desc)  
#get_summary_statistics(boston_x, data)

# Plot Histograms of Features
def generate_subplots(boston_x):
    fig, (axs1, axs2, axs3, axs4, axs5) = plt.subplots(5)    
    axs1.hist(boston_x[:,0 ])
    axs2.hist(boston_x[:,1])
    axs3.hist(boston_x[:,2])
    axs4.hist(boston_x[:,3])
    axs5.hist(boston_x[:,4])

def get_corrcoef_feature_v_target(boston_x, boston_y, boston_load):
    data = boston_load()
    features = data.get('feature_names')
    for num in range(0, len(boston_x[0,:])):
        coef = round(np.corrcoef(boston_x[:,num], boston_y)[1,0],2)
        print('Feature => {}  {}, , Coeff => {}'.format(features[num], num, coef))
               
#get_corrcoef_feature_v_target(boston_x, boston_y, boston_load)

def scatter_plot_feature_v_target(boston_x, boston_y):
    data = boston_load()
    features = data.get('feature_names')
    y_log = [m.log(y) for y in boston_y]
    
    try:
        for num in range(0, len(boston_x[0,:])):
            x_log = [m.log(int(x)) for x in boston_x[:,num]]
            plt.scatter(x_log, y_log)
            plt.title(features[num] + ' vs Median Home Price (log)')
            plt.show()
    except ValueError:
        for num in range(0, len(boston_x[0,:])):
            plt.scatter(boston_x[:,num], boston_y)
            plt.title(features[num] + ' vs Median Home Price')
            plt.show()

#scatter_plot_feature_v_target(boston_x, boston_y)


# SCATTER MATRIX ---------------------------------------

def olr_seaborn(boston_x, boston_y):
    print(tips.head())
    # Note: order controls the polynomial for the LR model. 
    sns.regplot(x= boston_x[:,5], y=boston_y, order=2)

def hist_seaborn(boston_x): 
    for num in range(0, len(boston_x[0, :])):
        sns.distplot(boston_x[:,num])
        plt.show()
    
def pairwise_plot():
    g = sns.PairGrid(df_boston)
    g.map(plt.scatter) 
    
pairwise_plot()
    




    