# -*- coding: utf-8 -*-
"""
Documentation: 
    Tutorial on Ordinary Least Squared Model
    np.newaxis:   
        Use to increase the dimension of the existing array by one or more dims. 
        https://medium.com/@ian.dzindo01/what-is-numpy-newaxis-and-when-to-use-it-8cb61c7ed6ae

"""

# IMPORT LIBRARIES ------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score



# IMPORT DATASETS -------------------------------
diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)
    
def print_dims():
    data = load_diabetes()
    dict_items = data.items()   
    print(dict_items)
    print('X value shape => {}'.format(diabetes_x.shape))

# Use only one feature
'[select all rows, create col vector, select col 2]'
diabetes_x = diabetes_x[:, np.newaxis, 2]

# Inspect data
def plot_inspect_data():    
    plt.scatter(diabetes_x, diabetes_y)
    plt.title('X value')

# Calculate Variance & Standard Deviation
x = np.array([[0,2], [1,1], [2,0]]).T
print(np.cov(x)[0,1]) 


# Split data into training/testing sets
diabetes_x_train = diabetes_x[:-20]
diabetes_x_test  = diabetes_x[-20:]

# Split Target into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test  = diabetes_y[-20:]

# Create Linear Regression Object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_x_train, diabetes_y_train)

# Make a prediction using the testing set
diabetes_y_pred = regr.predict(diabetes_x_test)

# The Coefficients
def print_results():
    print('Coefficients => {}'.format(regr.coef_))
    print('Intercept    => {}'.format(regr.intercept_))
    print('Mean Sqaured Error => {}'.format(mean_squared_error(
        diabetes_y_test, diabetes_y_pred)))
    print('Coefficient of determination => {}'.format(
        r2_score(diabetes_y_test, diabetes_y_pred)))

# Plot Outputs
'''Note:  The R2 value is likely very low because the spread of data
          is very wide'''
plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
plt.plot(diabetes_x_test, diabetes_y_pred, color='blue', linewidth=3)









