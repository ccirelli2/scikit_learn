#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Documentation: 
    Purpose: Train a decision tree using scikit learn. 
    url: https://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
'''

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

# Load Data
dir_data = r'/home/ccirelli2/Desktop/repositories/Scikit_Learn/data'
os.chdir(dir_data)
file_name = os.listdir(dir_data)[0]

df_data = pd.read_csv(file_name, 
                      sep=',', header=None)

# Investigate Data
'''Balance Scale data set consists of 5 attributes, 4 as feature attributes and 
   1 is the target attribute. We will try to build a classifier for predicting 
   the Class attribute. The index of target attribute is 1st.
   Taget = Column 0
   Features = Left-weight, left-distance, right-weight, right-distance
   Feature values: can range from 1-5'''

# Define X & Y Variables
X = df_data.iloc[:,1:]
y = df_data.iloc[:, 0]

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


# Vizualize Tree ------------------------------------
os.chdir(r'/home/ccirelli2/Desktop/repositories/Scikit_Learn/output')
dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data, 
                filled=True, rounded=True, 
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('balance_scale.png')
Image(graph.create_png())












