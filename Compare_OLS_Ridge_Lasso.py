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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

