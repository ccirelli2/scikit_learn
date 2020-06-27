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
pd.set_option('display.max.rows', 500)
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
sample = df_train.sample(10)
df_desc = df_train.describe()
continuous = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']
df_cont = df_train[continuous]


# Generate Correlation Matrix -------------------------------------------------
'With OLS we should always check if independent variables are correlated'
df_corr = df_train.corr(method='pearson')

def scatter_plot():
    plt.scatter(X_train['Item_MRP'], y_train)
    plt.show()

# Creating Dataset ------------------------------------------------------------
'''
Item_Identifier    Looks like a product key.  We can probably drop this
Item_Weight        Has Nan values
Item_Fat_Content   Is a categorical feature
Item_Type          Categorical Feature
Item_Visibility    Continuous / Nan Values
Item_MRP           Continuous
Outlet_Identifier  Categorical feature
Year               Ordinal
Outlet Size        Categorical
Outlet_Location_Type  Categorical
Outlet Type        Categorical
Item Outlet Sales  Continuous / Target Variable
'''

# Replace Nan Values W/ Mean
df_train['Item_Visibility'] = df_train['Item_Visibility'].replace(0, np.mean(df_train['Item_Visibility']))
df_train['Outlet_Establishment_Year'] = 2013 - df_train['Outlet_Establishment_Year']
df_train['Outlet_Size'].fillna('Small', inplace=True)  #small is one of the feature elements. 

# Create Dummy Variables
mylist = list(df_train.select_dtypes(include=['object']).columns)  #get colnames for categorical variables
dummies = pd.get_dummies(df_train[mylist], prefix=mylist)

# Drop Original Features
df_train.drop(mylist, axis=1, inplace=True)

# Create X Variable 
X = pd.concat([df_train, dummies], axis=1)


# Fit Model -------------------------------------------------------------------

# Instantiate Model 
lreg = LinearRegression()

# Isolate Independent Features
X.dropna(inplace=True)
y = X.loc[:, 'Item_Outlet_Sales']
X = X.drop('Item_Outlet_Sales', axis=1)

# Train / Test Split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size =0.3)

# Fit Model
lreg.fit(x_train, y_train)

# Predict
pred_cv = lreg.predict(x_cv)

# Calcualte MSE
mse = np.mean((pred_cv - y_cv)**2)

# R2
r2 = lreg.score(x_cv, y_cv)
print(r2)
















                                              
                                              







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    