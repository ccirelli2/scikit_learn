# Purpose------------------------------------------
'''
    Train, tune and vizualize nearest neighbor model
    Numpy axis = https://www.sharpsightlabs.com/blog/numpy-axes-explained/
'''


# Import Libraries ---------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
from sklearn import preprocessing

# Steps --------------------------------------------
''' 1.) Load Data
    2.) Exploratory Data Analysis
    3.) Standardize Data
    4.) Fit Model
    5.) Predict
    6.) Vizualize
'''

# Load Data -----------------------------------------
wine_data   = datasets.load_wine()
feature_names = wine_data['feature_names']
X, y        = datasets.load_wine(True)

# Standardize Data ----------------------------------

def create_df(X, y, feature_names):
    df = pd.DataFrame({})
    for n in range(0, len(feature_names)):
        feature = feature_names[n]
        data = X[:, n]
        mu = np.mean(data)
        sd = np.std(data)
        scaled = (data - mu)/sd
        df[feature] = scaled
    df['target'] = y
    # Return standardized data in a df
    return df

df_xy = create_df(X, y, feature_names)


def train_nn_model(df_xy, X, y, feature_names, scaled=True, plot=True, pprint=True):
    
    # Create Scaled X & Y Variables
    df_x = df_xy.drop('target', axis=1)
    df_y = df_xy['target']
    
    # Recall Values
    recall_scaled = []
    recall_not_scaled = []

    # Results Dataframe
    df_results = pd.DataFrame({})

    # Train & Test Split ------ -------------------------
    X_train, x_test, Y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=47)

    # Scaled Data
    for n in range(2, 10):
        nn = KNeighborsClassifier(n_neighbors=n)
        nn.fit(X_train, Y_train)
        nn_pred = nn.predict(x_test)
        class_report = classification_report(y_test, nn_pred)
        accuracy = accuracy_score(y_test, nn_pred)
        recall_scaled.append(accuracy)

    # Unscaled Data 
    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)

    # Iterate Over Range of Neighbors
    for n in range(2, 10):
        nn = KNeighborsClassifier(n_neighbors=n)
        nn.fit(X_train, Y_train)
        nn_pred = nn.predict(x_test)
        class_report = classification_report(y_test, nn_pred)
        accuracy = accuracy_score(y_test, nn_pred)
        recall_not_scaled.append(accuracy)
    
    # Populate Results DataFrame
    df_results['scaled'] = recall_scaled
    df_results['not-scaled'] = recall_not_scaled

    if plot == True:
        df_results.plot(kind='bar')
        plt.title('Scaled vs Not-Scaled Data')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Neighbors')
        plt.grid(True)
        plt.rcParams.update({'font.size': 22})
        plt.show()

    if pprint == True:
        print(recall)

train_nn_model(df_xy, X, y, feature_names, scaled=False)




