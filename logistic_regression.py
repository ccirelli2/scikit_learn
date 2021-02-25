# TUTORIAL:  SETTING UP LOGISTIC REGRESSION MODEL

'''DOCUMENTATION

Training_video:     https://www.youtube.com/
                    watch?v=WRaKnFB3SYQ&list=PLonlF40eS6nynU5ayxghbz2QpDsUAyCVF&index=8
                    https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
Dataset:            cancer (built-in)

Model:              Classification model.  Binary target 0/1. :


'C':                regularizes our dataset.  Higher C each datapoint will have to be
                    classified as accurately as possible.  Default is 1. 
                    Try C = .01 and 100.  Looking for a discrepency in the accuracy of the 
                    test and training set. 
'''

# Import Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Dataset
cancer = load_breast_cancer()

def train_log_regressor(cancer, C_value):
    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    stratify = cancer.target, 
                                                    random_state = 42)
    # Instantiate Model
    log_reg = LogisticRegression(C = C_value)
    log_reg.fit(x_train, y_train)

    # Print Scores
    print('Accuracy on the training subset: {}'.format(log_reg.score(x_train, y_train)))
    print('Accuracy on the test subset: {}'.format(log_reg.score(x_test, y_test)))











































