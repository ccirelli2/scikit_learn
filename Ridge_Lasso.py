# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:26:31 2020

@author: chris.cirelli
"""


# Import Libraries -----------------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# Load Data -------------------------------------------------------------------
dataset = sm.datasets.get_rdataset('Guerry', 'HistData')
data = dataset.data

# Inspect Data ----------------------------------------------------------------
def inspect_data(dataset):
    print(dataset.__doc__)
    print(dataset.data.head())

# Select Only Numberical Data Types -------------------------------------------
'Target = All_Crime'
data = data.select_dtypes(include=np.number)
data.drop('dept', axis=1, inplace=True)
data['All_Crime'] = data.loc[:, 'Crime_pers'] + data.loc[:, 'Crime_prop']

# Split X & Y
y = data.loc[:, 'All_Crime']
x = data.drop(['All_Crime', 'Crime_pers', 'Crime_prop'], axis=1)

# Split Train Test ------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, 
                                                    random_state = 123, shuffle=True)

# Fit OLS ---------------------------------------------------------------------
lreg = LinearRegression()
lreg.fit(x_train, y_train)
ols_pred = lreg.predict(x_test)

# Get Mean Squared Error ------------------------------------------------------
def get_metrics(y_test, pred, pprint):
    mse = mean_squared_error(y_test, pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, pred)
    coeffs = pd.Series(lreg.coef_)
    df_coeffs = pd.DataFrame({})
    df_coeffs['names'] = x_train.columns
    df_coeffs['coeffs'] = coeffs
    
    if pprint==True:
        print('MSE => {}'.format(mse))
        print('RMSE => {}'.format(rmse))
        print('R2 => {}'.format(r2))
        print('Coefficients => {}'.format(df_coeffs))
    
    return mse, rmse, r2    


# Fit Ridge Regression --------------------------------------------------------
from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha=0.5, normalize=True)
ridgeReg.fit(x_train, y_train)
ridge_pred = ridgeReg.predict(x_test)
#get_metrics(y_test, pred)

def compare_coeffs(x_train, ridgeReg, lreg, pprint, pplot):
    col_names = x_train.columns
    df = pd.DataFrame({})
    df['features'] = col_names
    df['ols_coeffs'] = pd.Series(lreg.coef_)
    df['ridge_coeffs'] = pd.Series(ridgeReg.coef_)
    if pprint == True:
        print(df)
    if pplot == True:
        df.plot(kind='bar')
    return df
#df_coeff_compare = compare_coeffs(x_train, ridgeReg, lreg, pprint=False, pplot=True)

def compare_metrics(y_test, ridge_pred, ols_pred):
    df = pd.DataFrame({})
    
    ols_mse, ols_rmse, ols_r2 = get_metrics(y_test, ols_pred, False)
    ridge_mse, ridge_rmse, ridge_r2 = get_metrics(y_test, ridge_pred, False)
    
    df['mse'] = [ols_mse, ridge_mse]
    df['rmse'] = [ols_rmse, ridge_rmse]
    df['r2'] = [ols_r2, ridge_r2]
    df['method'] = ['OLS', 'Ridge']
    
    print(df)
    
#compare_metrics(y_test, ridge_pred, ols_pred)
       
    
# Lasso -----------------------------------------------------------------------
from sklearn.linear_model import Lasso
lassoReg = Lasso()
lassoReg.fit(x_train, y_train) 
lasso_pred = lassoReg.predict(x_test)

get_metrics(y_test, lasso_pred, pprint=True)   
    
df = compare_metrics(y_test, lasso_pred, ridge_pred)

    
compare_coeffs(x_train, lassoReg, ridgeReg, pprint=False, pplot=True)    
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    








    



















