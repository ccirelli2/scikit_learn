#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial on how to train a random forecast classifier
url:  https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
data:  https://www.ncdc.noaa.gov/cdo-web/

Problem:  predict max temperature tomorrow using past weather data. 
Prediction: Regression based as the target is continous. 

Workflow:
    State the question and determine required data
    Acquire the data in an accessible format
    Identify and correct missing data points/anomalies as required
    Prepare the data for the machine learning model
    Establish a baseline model that you aim to exceed
    Train the model on the training data
    Make predictions on the test data
    Compare predictions to the known test set targets and calculate performance metrics
    If performance is not satisfactory, adjust the model, acquire more data, or try a different modeling technique
    Interpret model and report results visually and numerically

Data
    Following are explanations of the columns:
    year: 2016 for all data points
    month: number for month of the year
    day: number for day of the year
    week: day of the week as a character string
    temp_2: max temperature 2 days prior
    temp_1: max temperature 1 day prior
    average: historical average max temperature
    actual: max temperature measurement
    friend: your friend’s prediction, a random number between 20 below the average and 20 above the average

Explanation of Random Forest Training
    url: https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
    1.) Random sampling of training data points when building trees
    When training, each tree in a random forest learns from a random sample of 
    the data points. The samples are drawn with replacement, known as bootstrapping, 
    which means that some samples will be used multiple times in a single tree. The idea 
    is that by training each tree on different samples, although each tree might have high 
    variance with respect to a particular set of the training data, overall, the entire forest 
    will have lower variance but not at the cost of increasing the bias.
    
    At test time, predictions are made by averaging the predictions of each decision tree. 
    This procedure of training each individual learner on different bootstrapped subsets of 
    the data and then averaging the predictions is known as bagging.
    
    2.) Random subsets of features considered when splitting nodes
    The other main concept in the random forest is that only a subset of all the 
    features are considered for splitting each node in each decision tree. Generally 
    this is set to sqrt(n_features) for classification meaning that if there are 16 
    features, at each node in each tree, only 4 random features will be considered for 
    splitting the node. (The random forest can also be trained considering all the features 
    at every node as is common in regression. These options can be controlled in 
    the Scikit-Learn Random Forest implementation).

"""

# Import Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot

# Load Data
data_dir = r'/home/ccirelli2/Desktop/repositories/Scikit_Learn/data'
out_dir = r'/home/ccirelli2/Desktop/repositories/Scikit_Learn/output'
 
filename = r'temps.csv'
df_temps = pd.read_csv(data_dir + '/' + filename)


# Basic Data Inspection 
'Note: There are only 348 days '
def data_insepction(df):
    print('Df dimensions => {}\n'.format(df.shape))
    print('Features => {}\n'.format(df.columns))
    print('Head')
    print(df.head(), '\n')
    print('Description')
    print(df.describe())
    print('Count Null Values => {}\n'.format(df.isnull().values.sum()))
    
    
def data_line_plots(df):
    # Limit dataframe to the feature
    df_limited = df[['temp_2', 'temp_1', 'average', 'actual',
       'forecast_noaa', 'forecast_acc', 'forecast_under', 'friend']]
    # Get column names
    cols = df_limited.columns
    # Iterate over each column and plot value vs day
    for col in cols:
        df_col = df_limited[col]
        plt.plot(df_col)
        plt.title(col)
        plt.show()
        

def data_scatter_plots(df):
    # Limit dataframe to the feature
    df_limited = df[['temp_2', 'temp_1', 'average', 'actual',
       'forecast_noaa', 'forecast_acc', 'forecast_under', 'friend']]
    # Get column names
    cols = df_limited.columns
    # Iterate over each column and plot value vs actual
    for col in cols:
        df_col = df_limited[col]
        plt.scatter(x=df_col, y=df_limited['actual'])
        plt.title(col)
        plt.show()

def inspect_features(df):
    '''Purpose: Inspect datatype of features'''
    ddict ={}
    for col in zip(df.columns, range(0, len(df.columns))):
        datapoint = df.iloc[:1, col[1]]
        ddict[col[0]] = type(datapoint[0])
    # Print Dictionary
    print(ddict)


# Observations     
'''Note: we have integers, float and string feature values.
    We will need to properly prepare out data before utilizing 
    a machine learning model
'''
    

# Data Preparation ------------------------------------------------
''' 1.) How do we handle the floats vs ints?
    2.) How do we handle the character values?
'''


# Step 1 - One Hot Encode the Day of Week
''' Feature name = 'week'
'''

# Using Pandas
''' Note:  The function only applies to character feature values. 
           The features comprised of integers and or floats were not
           adjusted. 
'''
features = pd.get_dummies(df_temps)

# Display all columns when calling head()
pd.set_option('display.max_columns', len(features.columns))


# Split Features & Target (tutorial says to use numpy arrays)
X = features.drop('actual', axis=1)
feature_names = X.columns  # save column names
y = features['actual']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 100)

# Ensure that our train test split worked correctly
def get_shape_splits():
    for split in [X_train, X_test, y_train, y_test]:
        print(split.shape)
    

# *** Calculate a baseline error for our prediction
''' What would be our error simply using the average
    as our prediction?  This should be the baseline
    for our model to try to beat.
    Note:  The autor chose to use degrees as opposed to
           mse. 
'''
avg_tmp = X.iloc[:,5]
baseline_error_degrees = round(np.mean(abs(y - avg_tmp)),2)



# Train Model -----------------------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

#print(rf.get_params())
#print(rf.score(X_train, y_train))
#print('Random Forest Parameters', rf)

# Generate Prediction ---------------------------------------
'''Remember, you are generating a prediction from the test data
   and checking it relative to the actual y_test data. 
'''
rf_pred = rf.predict(X_test)




# Results ---------------------------------------------------
rf_error_degrees = round(np.mean(abs(y_test - rf_pred)), 2)

print('Baseline error => {} vs Actual => {}'.format(
        baseline_error_degrees, rf_error_degrees))


# Mean Percentage Error
mape = 100 - np.mean((100 * (abs(y_test - rf_pred)/ y_test)))
print('Mean Average Prediction Error => {}'.format(mape))



# Vizualize Feature Importance ---------------------------------
importance = rf.feature_importances_

df_feature_imp = pd.DataFrame({})
df_feature_imp['Feature'] = feature_names
df_feature_imp['Imp'] = importance
df_feature_imp.sort_values('Imp').plot(kind='bar')


# Vizualize Tree ------------------------------------------------
'Note:  This is a fully grown tree that probably requires pruning'
def viz_tree():
    atree = rf.estimators_[5]
    export_graphviz(atree, out_file= out_dir + '/' + 'random_forest_plot.dot', 
            feature_names=feature_names, rounded=True, precision=1)




