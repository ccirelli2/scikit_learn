# -*- coding: utf-8 -*-
"""
Desc   Tutorial on Ridge & Lasso Regression
Source https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/
       https://stackoverflow.com/questions/26319259/how-to-get-a-regression-summary-in-python-scikit-like-r-does
"""


# Import Libraries ------------------------------------------------------------
import pandas as pd
from pandas.plotting import scatter_matrix
pd.set_option('display.max.columns', None)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Define Directories ----------------------------------------------------------
dir_data = r'C:\Users\chris.cirelli\Desktop\repositories\scikit_learn\data'

# Load Data -------------------------------------------------------------------
df_train = pd.read_csv(dir_data + '/' + 'groceries_train.csv')
df_test = pd.read_csv(dir_data + '/' + 'groceries_test.csv')


# Inspect Data set ------------------------------------------------------------
col_names = df_train.columns
sample = df_train.sample(5)
continuous = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']
df_cont = df_train[continuous]


# Generate Correlation Matrix -------------------------------------------------
'With OLS we should always check if independent variables are correlated'
df_corr = df_train.corr(method='pearson')
#scatter_matrix(df_cont)


# Split Data Into X Y ---------------------------------------------------------
y = df_train.Item_Outlet_Sales
X = df_train.loc[:, ['Outlet_Establishment_Year', 'Item_MRP']]
X_train, x_cv, y_train, y_cv = train_test_split(X, y, 
                                                test_size=.30, 
                                                random_state = 123)

# Generate Scatter plot
def scatter_plot():
    plt.scatter(X_train['Item_MRP'], y_train)
    plt.show()


# Linear Regression -----------------------------------------------------------
lreg = LinearRegression()

# Fit Model 
'Note:  Sklearn has no summary output like R'
lreg.fit(X_train, y_train)

# Generate Prediction 
pred = lreg.predict(x_cv)

# R2
r2 = lreg.score(x_cv, y_cv)

# Results
def get_lreg_results(X_train, x_cv, y_train, y_cv, lreg, pred):

    # Calculate MSE
    mse = np.mean(pred - y_cv)**2
    
    # Get Coefficients
    coeff = pd.DataFrame(X_train.columns)
    coeff['Coefficient Estimate'] = pd.Series(lreg.coef_)
    
    # Explained Variance
    explained_var = metrics.explained_variance_score(y_cv, pred)
    
    # Mean Absolute Error
    mean_abs_error = metrics.mean_absolute_error(y_cv, pred)
    
    # Mean Squared Log Error
    mean_sqared_log_error = metrics.mean_squared_log_error(y_cv, pred)
    
    # R^2
    r2 = metrics.r2_score(y_cv, pred)
    
    # print results
    print('MSE => {}'.format(mse))
    print('Coefficients => {}'.format(coeff))
    print('Explained Variance => {}'.format(explained_var))
    print('Mean Absolute Error => {}'.format(mean_abs_error))
    print('Mean Squared Log Error => {}'.format(mean_sqared_log_error))
    print('R^2 => {}'.format(r2))

#get_lreg_results(X_train, x_cv, y_train, y_cv, lreg, pred)
    
    
# Get Summary From Python Stats Model
def get_summary_python_stats_model():
    X_train = X_train['Item_MRP']
    y_train = y_train.values
    model = sm.OLS(X_train, y_train)
    results = model.fit()
    print(results.summary())




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    