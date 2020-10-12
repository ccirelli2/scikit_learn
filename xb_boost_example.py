# -*- coding: utf-8 -*-
"""
Reference : https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
"""

# Import Libraries ------------------------------------------------------------
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load Dataset ----------------------------------------------------------------
data = loadtxt(r'C:\Users\chris.cirelli\Desktop\repositories\scikit_learn\data\pima-indians-diabetes.data.csv',
               delimiter=',')


# split data into X and y
X = data[:,0:8]
Y = data[:,8]

# split data into X and y
X = data[:,0:8]
Y = data[:,8]


# split data into X and y
X = data[:,0:8]
Y = data[:,8]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)