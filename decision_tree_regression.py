# DOCUMENTATION ---------------------------------------
''' Tutorial on how to train a decision tree regression model in scikit learn
    Source:  https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py

    Structure:  given an x input value, predict y

'''



# IMPORT LIBRARIES ------------------------------------
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import random as rnd

# Create a Random Dataset
'Numpy RandomState: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html'
rng = np.random.RandomState(1)
x   = np.sort(5 * rng.rand(80,1), axis=0) # generates a np.array of dim 80x1 
y   = np.sin(x).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


# Train / Test Split
train_split = int(len(x) * 0.8)
x_train = x[: train_split]
x_test  = x[train_split:]
y_train = y[: train_split]
y_test  = y[train_split:]



# Fit Regression Model

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x,y)
regr_2.fit(x,y)

# Generate Prediction
y_pred1 = regr_1.predict(x_test)
y_pred2 = regr_2.predict(x_test)


# Calculate MSE
mse1 = sum((y_test - y_pred1)**2)
mse2 = sum((y_test - y_pred2)**2)

print(mse1)
print(mse2)


# Plot Results
def results(x,y,y_pred):
    plt.figure(figsize=(20,20))
    plt.plot(x_test, y_test)
    plt.plot(x_test, y_pred)
    plt.show()
    

# ***** NOTE WE NEED TO REVIEW THE SCORING FUNCTION
    
    



