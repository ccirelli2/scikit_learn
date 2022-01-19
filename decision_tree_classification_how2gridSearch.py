#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Documentation: 
    Purpose: Train a decision tree using scikit learn. 
    url: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
'''

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load Data
dir_data = r'/home/ccirelli2/Desktop/repositories/Scikit_Learn/data'
os.chdir(dir_data)
file_name = os.listdir(dir_data)[0]

df_data = pd.read_csv(file_name, 
                      sep=',', header=None)

# GridSearchCV -----------------------------------------------------
'''Stands for Grid Search Cross Validation.
   Steps: 
       - load data
       - define parameter objects
       - define dictionary object of parameters
       - instantiate your model
       - pass model to GridSearchCV along with parameters
       - Fit Model
'''

# Step 1: Define features & target variables
X = df_data.iloc[:,1:]
y = df_data.iloc[:, 0]

# Step 2: Create lists of parameters for your model 
criterion = ['gini', 'entropy']
max_depth = [x for x in range(2, 12, 2)]
min_samples_split = [x for x in range(2, 12, 2)]
min_samples_leaf = [x for x in range(1,10)]

# Step 3: Create a dictionary with the parameter names that coincide with model
parameters = dict(criterion=criterion, max_depth=max_depth, 
                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

# Instantiate Model
dtc = DecisionTreeClassifier(random_state=100)

# Create Grid Search Object, Pass Model & Parameter Grid
grid = GridSearchCV(dtc, parameters, cv=10, scoring='accuracy')

# Fit Model 
grid.fit(X, y)

# Print Results
best_estimators = grid.best_estimator_.get_params()
best_score = grid.best_score_
best_params = grid.best_params_

print('Best Estimators => {}\n'.format(best_estimators))
print('Best scores => {}\n'.format(best_score))
print('Best parameters => {}\n'.format(best_params))


# Manual Fit -----------------------------------------------------------

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 100)

# Instantiate Model
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5, 
                                  min_samples_split=2, random_state=100, splitter='best')
# Fit Model
clf_gini.fit(X_train, y_train)

# Generate Prediction
clf_gini_pred = clf_gini.predict(X_test)

# Calculate Accuracy Score
'''Accuracy = ratio of correctly predicted target values vs all values'''
clf_gini_accuracy = round(accuracy_score(y_test, clf_gini_pred) * 100, 2)
print(clf_gini_accuracy)





