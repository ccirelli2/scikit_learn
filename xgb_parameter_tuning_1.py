# -*- coding: utf-8 -*-
"""
###############################################################################
# XGBoost Parameter Tuning
###############################################################################

Created on Fri Jan 22 17:23:10 2021
@author: chris.cirelli

ref: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

"""

# Import Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt


###############################################################################

###############################################################################

