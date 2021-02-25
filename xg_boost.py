# -*- coding: utf-8 -*-
"""
Ref1 : https://www.datacamp.com/community/tutorials/xgboost-in-python
Ref2 : https://xgboost.readthedocs.io/en/latest/   #XGboost docs
cv   : https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
hyper parameter tuning grid search
    https://towardsdatascience.com/beyond-grid-search-hypercharge-hyperparameter-tuning-for-xgboost-7c78f7a2929d
    https://towardsdatascience.com/doing-xgboost-hyper-parameter-tuning-the-smart-way-part-1-of-2-f6d255a45dde
    https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
    
Data :
    1.) Categorical Variables:
        - Need to be one hot encoded
    2.) Nan or Null Values
        - XGboost can handle them yet it is prudent to address them explicitly
        when fitting a model
    3.) Input
        - Must be in the form of a Dmatrix. 

Parameters : 
    ref : https://xgboost.readthedocs.io/en/latest/parameter.html
    1.) Global Parameters
        xgb.set.config()
        ex: xgb.config_context(verbosity=)
            0=silent, 1=warning, 2=info, 3=debug

General Parameters:
    1.) booster (default = gbtree)
        - gbtree and dar used tree based models while gblinear uses linear
        functions.

Learning Paremters (Not exhaustive):
    1.) learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
    2.) max_depth: determines how deeply each tree is allowed to grow during
        any boosting round.
    3.) subsample: percentage of samples used per tree. Low value can lead to
        underfitting.
    4.) colsample_bytree: percentage of features used per tree. High value can
        lead to overfitting.
    5.) n_estimators: number of trees you want to build.
    6.) objective: determines the loss function to be used like reg:linear for
    regression problems, reg:logistic for classification problems with only
    decision, binary:logistic for classification problems with probability.

Regularization Parameters
    1.) gamma: controls whether a given node will split based on the expected
        reduction in loss after the split. A higher value leads to fewer splits.
        Supported only for tree-based learners.
    2.) alpha: L1 regularization on leaf weights. A large value leads to more regularization.
    3.) lambda: L2 regularization on leaf weights and is smoother than L1 regularization.


Cross Validation :
    1.) Built into Xgboost.  All you need to do is supply the nfolds parameter.
    Other Parameters
    2.) num_boost_round: denotes the number of trees you build (analogous to n_estimators)
    3.) metrics: tells the evaluation metrics to be watched during CV
    4.) as_pandas: to return the results in a pandas DataFrame.
    5.) early_stopping_rounds: finishes training of the model early if the
    hold-out metric ("rmse" in our case) does not improve for a given
    number of rounds.
    6.) seed: for reproducibility of results.

    
Questions :
    1.) What does it mean that XGboost gives more weight to misclassified
    observations?

"""
###############################################################################
# Import Python Libraries
###############################################################################
import logging; logging.basicConfig(level=logging.INFO)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

###############################################################################
# Data
###############################################################################
boston = load_boston()
col_names = boston['feature_names']
data = pd.DataFrame(boston['data'], columns=col_names)
data_desc = boston['DESCR']
# Data Inspection
logging.debug(f'---- data shape => {data.shape}')
logging.debug(f'--- feature names => {col_names}')
logging.debug(f'---- descirption \n {data_desc}')

###############################################################################
# Data Pre-processing
###############################################################################
X = data.drop('CRIM', axis=1)
y = data['CRIM'].values

# Get Dmatrix
data_dmatrix = xgb.DMatrix(data=X, label=y)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=123)

###############################################################################
# Regression Problems
###############################################################################


# Fit XGBRegressor
def xg_boost_regression(X_train, X_test, y_train, y_test, pplot_tree,
                        pplot_imp):
    """
    Fit XGBRegressor to regression problem.
    
    objective function : options include 'reg:linear', 'reg:squaredlogerror'

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    pplot_tree : TYPE
        DESCRIPTION.
    pplot_imp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Instantiate Model
    model = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3,
                             learning_rate=0.1, max_depth=5, alpha=10,
                             n_estimators=10)
    # Fit To Training Data
    model.fit(X_train, y_train)
    # Predict X Test
    y_preds = model.predict(X_test)
    # Get RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    # Results
    logging.info(f'---- rmse => {rmse}')
    # Plot
    if pplot_tree:
        xgb.plot_tree(model, num_trees=0)
        plt.show()
    if pplot_imp:
        xgb.plot_importance(model)
        plt.show()

"""
xg_boost_regression(X_train, X_test, y_train, y_test, pplot_tree=True,
                    pplot_imp=True)
"""

hyper_params = {"objective": "reg:squarederror", "colsample_bytree": 0.3,
                "max_depth": 5, "alpha": 10}

def xg_boost_regression_cv(
        X_train, X_test, y_train, y_test, params, n_folds, num_boost_round,
        early_stopping_rounds, metrics, as_pandas, seed):
    
    # Create Dmatrix from training data
    dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    # Pass Parameters to xgb.cv()
    cv_results = xgb.cv(dtrain=dmatrix, params=params,
                        nfold=3, num_boost_round=10,early_stopping_rounds=10,
                        metrics="rmse", as_pandas=True, seed=123)
    best_params = best_params[0]
    # Log Results
    logging.info(f'---- cv results => \n\n{cv_results.head()}\n\n')
    logging.info(f'---- final rmse results => {cv_results["test-rmse-mean"].tail(1)}')
    

"""
xg_boost_regression_cv(
        X_train, X_test, y_train, y_test, params=hyper_params, n_folds=3,
        num_boost_round=50, early_stopping_rounds=10, metrics='rmse',
        as_pandas=True, seed=123)
"""                    

params = {
    'objective':'Binary',
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear',
}

def xg_boost_best_params(X_train, X_test, y_train, y_test, params,
                         num_boost_round):
    
    # Create DMatrix for Train & Test Data
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)
    # Instantiate Model with Parameters
    model = xgb.train(params, dtrain, num_boost_round,
                      evals=[(dtest, "Test")], early_stopping_rounds=10)
    # Return Best Results
    best_score = model.best_score
    best_iter = model.best_iteration
    # Logging
    logging.info(f'---- best score => {best_score}')
    logging.info(f'---- best iter => {best_iter}')


xg_boost_best_params(X_train, X_test, y_train, y_test, params,
                     num_boost_round=10)


"""
Continue 
https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f

Shap & Lime
https://blog.dominodatalab.com/shap-lime-python-libraries-part-2-using-shap-lime/

"""


###############################################################################
# XGBoost Parameter Tuning
###############################################################################
'ref: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/'










