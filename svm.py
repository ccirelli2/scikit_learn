# PURPOSE -----------------------------------------------------
''' Description: Train a Support Vector Machine
                Predict class of wine maker
                Compare to RandomForest & K nearest neighbor algorithms
    URL:         https://scikit-learn.org/stable/modules/svm.html
    Approach:   Bayes classification uses a probabilistic model to determine
                each class label, which is an example of a generative model. 
                SVM is a discriminative classification model, where we fine a line or
                curve that divides the classes from each other. 



'''


# LIBRARIES ---------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm
from sklearn.datasets import make_blobs
from mpl_toolkits import mplot3d

# Example from Tutorial -----------------------------------------
def plot_ex_scatter():
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
    plt.scatter(X[:,0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.show()


# Adding Dimensions to Data

def plot_3D(elev=30, azim=30):
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
    r = np.exp(-(X**2).sum(1))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()


# IMPORT WINE DATASETS ------------------------------------------
wine_data = datasets.load_wine()
feature_names = wine_data['feature_names']
X, y = datasets.load_wine(True)

# Scale Data
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
    # Return standardized data in a
    return df

df_xy = create_df(X, y, feature_names)
df_x = df_xy.drop('target', axis=1)
df_y = df_xy['target']
X_train, x_test, Y_train, y_test = train_test_split(df_x, df_y, test_size = 0.3, random_state=47)


# INSTANTIATE & FIT MODEL -------------------------------------

clf = svm.SVC(kernel='rbf', C=1.0)
clf.fit(X_train, Y_train)
clf_pred = clf.predict(x_test)

class_report = classification_report(y_test, clf_pred) 
accuracy = accuracy_score(y_test, clf_pred)
print(clf)
print(class_report)
print(accuracy)
#print('Support Vectors => {}'.format(clf.support_vectors_))








