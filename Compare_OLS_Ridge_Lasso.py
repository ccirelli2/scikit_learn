# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 06:57:08 2020

@author: chris.cirelli
"""


# Import Libraries -----------------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import math


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# Load Data Set ---------------------------------------------------------------
crime = sm.datasets.longley.load_pandas()
crime_data = crime.data


# Generate Covariance Matrix --------------------------------------------------
data_matrix = crime_data.copy()
data_matrix.drop('YEAR', axis=1, inplace=True)
scaler1 = StandardScaler()
scaler1.fit(data_matrix)
data_matrix_transformed = scaler1.transform(data_matrix)

df_corr_matrix = pd.DataFrame(data_matrix_transformed, 
                              columns=data_matrix.columns).corr()




# Prepare Data set ------------------------------------------------------------

# Scale X Features & Target Variable
'Drop Year As It Needs to Be One Hot Encoded'
crime_data_copy = crime_data.copy()
crime_data_copy.drop('YEAR', axis = 1, inplace=True)

scaler = StandardScaler()
scaler.fit(crime_data_copy)
crime_data_transformed = scaler.transform(crime_data_copy)

# Return DataFrame with Column Names
df = pd.DataFrame(crime_data_copy, columns=crime_data_copy.columns)
df['Year'] = [str(int(x)) for x in crime_data.loc[:, 'YEAR']]

# Define X & Y
Y = df.loc[ : ,'TOTEMP']
X = df.drop('TOTEMP', axis=1)

# One Hot Encode Using Pandas Dummies 
X = pd.get_dummies(X, prefix='_', columns=['Year'])


# Train Test Split ------------------------------------------------------------
x_train, x_cv, y_train, y_cv = train_test_split(X, Y, test_size = 0.3)


# Try Iterative Approach to Feature Selection ---------------------------------
def ols_iterative_features(x_train, x_cv, y_train, y_cv, col):
    
    x_train = np.array(x_train.loc[:, col]).reshape(-1,1)
    x_cv = np.array(x_cv.loc[:, col]).reshape(-1,1)

    lreg = LinearRegression()
    lreg.fit(x_train, y_train)  
    pred = lreg.predict(x_cv)
    mse = mean_squared_error(y_cv, pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_cv, pred)
    
    print('Feature => {}'.format(col))
    print('MSE => {}'.format(mse))
    print('RMSE => {}'.format(rmse))
    print('R2 => {}\n'.format(r2))
        

#for col in x_train.columns:
#    ols_iterative_features(x_train, x_cv, y_train, y_cv, col)



# Fit Models ------------------------------------------------------------------

def fit_regression(reg_type, x_train, x_cv, y_train, y_cv, alpha=0.5, 
                   pprint=True):

    if reg_type == 'OLS' or reg_type == 'ols':    
        lreg = LinearRegression()
        lreg.fit(x_train, y_train)  
        pred = lreg.predict(x_cv)

    elif reg_type == 'Ridge' or reg_type == 'ridge':
        lreg = Ridge(alpha=alpha, normalize=True)
        lreg.fit(x_train, y_train)
        pred = lreg.predict(x_cv)
        
    elif reg_type == 'Lasso' or reg_type == 'lasso':
        lreg = Lasso(tol=0.00001, max_iter=1000000000)
        lreg.fit(x_train, y_train) 
        pred = lreg.predict(x_cv)

    else:
        print('Regression type must be ols, ridge or lasso.  Please try again')

    # Calculate Metrics
    mse = mean_squared_error(y_cv, pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_cv, pred)
    coeffs = pd.Series(lreg.coef_)
    df = pd.DataFrame({})
    df['names'] = x_train.columns
    df['coeffs'] = coeffs

    if pprint==True:
        print('MSE => {}'.format(mse))
        print('RMSE => {}'.format(rmse))
        print('R2 => {}'.format(r2))
        print('Coefficients \n')
        #print(df)

    # Return values
    return mse, rmse, r2, df


# Results ---------------------------------------------------------------------
#mse_ols, rmse_ols, r2_ols, df_ols = fit_regression('ols', x_train, x_cv, y_train, y_cv, pprint=False)
#mse_ridge, rmse_ridge, r2_ridge, df_ridge = fit_regression('Ridge', x_train, x_cv, y_train, y_cv, alpha = 0.8, pprint=False)
#mse_lasso, rmse_lasso, r2_lasso, df_lasso = fit_regression('Lasso', x_train, x_cv, y_train, y_cv, pprint=False)


# Compare RMSE & R2
def compare_metrics():
    df_comp = pd.DataFrame({})
    df_comp['model'] = ['ols', 'ridge', 'lasso']
    df_comp['r2']   = [r2_ols, r2_ridge, r2_lasso]
    df_comp['rmse']  = [rmse_ols, rmse_ridge, rmse_lasso]    
    print(df_comp)

# Plot Coefficients -----------------------------------------------------------
def plot_coeffs():
    df = df_ols.merge(df_ridge, left_on='names', right_on='names').merge(df_lasso, left_on='names', right_on='names')
    df.rename(columns={'names':'names', 'coeffs_x':'OLS', 'coeffs_y':'Ridge', 'coeffs':'Lasso'}, 
              inplace=True)
    
    df_ols.plot(kind='bar', title='OLS Coefficients')
    df_ridge.plot(kind='bar', title='Ridge Coefficients')
    df_lasso.plot(kind='bar', title='Lasso Coefficients')




























