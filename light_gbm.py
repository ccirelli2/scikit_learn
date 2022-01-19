# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:28:02 2021
@author: chris.cirelli

References:
    Docs: https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
    CV: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.cv.html
    Python API: https://lightgbm.readthedocs.io/en/latest/Python-API.html
    Data: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    Tutorials:
        https://sefiks.com/2018/10/13/a-gentle-introduction-to-lightgbm-for-applied-machine-learning/
        https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
        *https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
        https://www.kaggle.com/ezietsman/simple-python-lightgbm-example
    Parameter Tuning:
        https://neptune.ai/blog/lightgbm-parameters-guide
        https://towardsdatascience.com/the-10-best-new-features-in-scikit-learn-0-24-f45e49b6741b
        https://medium.com/@sergei740/hyperparameter-tuning-lightgbm-using-random-grid-search-dc11c2f8c805
        https://towardsdatascience.com/the-10-best-new-features-in-scikit-learn-0-24-f45e49b6741b
    Offset:
        https://towardsdatascience.com/offsetting-the-model-logic-to-implementation-7e333bc25798
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Dataset.html
"""

###############################################################################
# Import Libraries
###############################################################################
import logging
import sys
import os
import numpy as np
import pandas as pd
from numpy import mean, std
from matplotlib import pyplot as plt

import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split


from sklearn.model_selection import GridSearchCV
#from sklearn.experimental import enable_halving_search_cv  
#from sklearn.model_selection import HalvingGridSearchCV
#from sklearn.model_selection import HalvingRandomSearchCV


###############################################################################
# Library Settings
###############################################################################
logging.basicConfig(level=logging.INFO)


###############################################################################
# Directories
###############################################################################
dir_repo=r'C:\Users\chris.cirelli\Desktop\repositories\scikit_learn'
dir_data=os.path.join(dir_repo, 'data')
sys.path.append(dir_data)

###############################################################################
# Create Dataset
###############################################################################

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2,
                           n_redundant=3, n_classes=2, random_state=1)


###############################################################################
# Data Transformation
###############################################################################
"""light gbm requires that categorical features get converted to integers, but
   does not require one hot encoding"""

# Train test split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33,
                                                  random_state=123)

###############################################################################
# Train Light GBM Model For Classification
###############################################################################
""" Note that lgbm requires that we pass parameters in as a dictionary.
"""
# Define parameter
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'                # boosting type
params['objective'] = 'binary'                  # classification
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 20
params['max_depth'] = 10


def lgb_binary(X_train, y_train, X_test, y_test, params, num_rounds):
    # Convert dataset to lgb dataset
    d_train=lgb.Dataset(X_train, label=y_train)
    # Train Model
    clf=lgb.train(params, d_train, num_boost_round=num_rounds)    
    # Predict
    yhat=clf.predict(X_test)
    # Convert Probabilities into binary variables
    y_hat=list(map(lambda x: 1 if x >= 0.5 else 0, yhat))
    # Get Confusion Matrix
    cm = confusion_matrix(y_test, y_hat)
    # Get Accuracy Score
    score=accuracy_score(y_test, y_hat)
    print(cm, '\n')
    print(score)
    # Plotting
    ax = lgb.plot_importance(clf, max_num_features=10)
    plt.show()
    ax = lgb.plot_tree(clf) 
    plt.show()
lgb_binary(X_train, y_train, X_test, y_test, params, num_rounds=500)


    
def lgb_binary_cv(X_train, y_train, X_test, y_test, params, num_rounds):
    # Convert dataset to lgb dataset
    d_train=lgb.Dataset(X_train, label=y_train)
    cv_results=lgb.cv(params, d_train, num_boost_round=100, nfold=10)
    print(cv_results)


###############################################################################
# Train Light GBM Model For Classification - Grid Search Approaches
###############################################################################

gridParams = {}
gridParams['learning_rate']=    [0.1, 0.01, 0.001, 0.001]
gridParams['boosting_type'] =   ['gbdt', 'dart']
gridParams['objective'] =       ['binary']                  
gridParams['metric'] =          ['binary_logloss']
gridParams['n_estimators'] =    [100, 500, 1000]
"""
gridParams['sub_feature'] =     [0.5]
gridParams['num_leaves'] =      [5, 10, 15, 20]
gridParams['max_depth'] =       [5, 10, 20, 30]

gridParams['colsample_bytree'] = [0.65]
gridParams['subsample'] =       [0.7, 0.8]
gridParams['min_data'] =        [20]
"""

def lgb_gridsearch(X_train, y_train, X_test, y_test, gridParams):
    # Instantiate Model
    lgb=LGBMClassifier(verbosity=-1)
    # Pass Model & Params to GridSearch Object
    grid=HalvingRandomSearchCV(lgb, gridParams)
    # Fit GridSearch Object on Training data
    grid.fit(X_train, y_train)
    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)
    # Return grid boject
    return grid

#grid=lgb_gridsearch(X_train, y_train, X_test, y_test, gridParams)



# Utilize Grid to Update Parameter Object
"""
params['learning_rate'] = grid.best_params_['learning_rate']
params['boosting_type'] = grid.best_params_['boosting_type']
params['objective'] = grid.best_params_['objective']                 
params['metric'] = grid.best_params_['metric']
params['sub_feature'] = grid.best_params_['sub_feature']
params['num_leaves'] = grid.best_params_['num_leaves']
params['min_data'] = grid.best_params_['min_data']
params['max_depth'] = grid.best_params_['max_depth']
params['n_estimators'] = grid.best_params_['n_estimators']
params['subsample'] = grid.best_params_['subsample']
"""



















































###############################################################################
# Train Light GBM Model For Classification - With Parameter Tuning
###############################################################################
"""
Following set of practices can be used to improve your model efficiency.

num_leaves: This is the main parameter to control the complexity of the
    tree model. Ideally, the value of num_leaves should be less than or equal to
    2^(max_depth). Value more than this will result in overfitting.
min_data_in_leaf: Setting it to a large value can avoid growing too deep a
    tree, but may cause under-fitting. In practice, setting it to hundreds or
    thousands is enough for a large dataset.

max_depth: You also can use max_depth to limit the tree depth explicitly.

For Faster Speed:
    Use bagging by setting bagging_fraction and bagging_freq
    Use feature sub-sampling by setting feature_fraction
    Use small max_bin
    Use save_binary to speed up data loading in future learning
    Use parallel learning, refer to parallel learning guide.

For better accuracy:
    Use large max_bin (may be slower)
    Use small learning_rate with large num_iterations
    Use large num_leaves(may cause over-fitting)
    Use bigger training data
    Try dart
    Try to use categorical feature directly

To deal with over-fitting:
    Use small max_bin
    Use small num_leaves
    Use min_data_in_leaf and min_sum_hessian_in_leaf
    Use bagging by set bagging_fraction and bagging_freq
    Use feature sub-sampling by set feature_fraction
    Use bigger training data
    Try lambda_l1, lambda_l2 and min_gain_to_split to regularization
    Try max_depth to avoid growing deep tree

"""








def lgb_class(X_train, y_train):
    # Instantiate Model
    model=LGBMClassifier()
    # evaluate the model
    model = LGBMClassifier()
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print(f'---- Accuracy => {mean(n_scores)}, Stdv => {std(n_scores)}')
    
    # Fit Single Model
    model.fit(X_train, y_train)
    
    lgb.plot_importance(model,max_features=10)


    
###############################################################################
# Train Light GBM Model For Regression
###############################################################################

def lbg_regression(X_train, y_train):
    model=LGBMRegressor()
    model.fit(X_train, y_train)
    y_hat=model.predict([[2, 85, 85, 1]])
    print(y_hat)
    





