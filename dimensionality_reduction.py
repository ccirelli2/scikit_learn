# DOCUMENTATION -------------------------------------------
'''
  Tutorial:  PCA, Kernel PCA, LDA, SVM
  
  Ref: https://towardsdatascience.com/dimension-reduction-techniques-with-python-f36ca7009e5c
  Ref Kernel PCA https://www.geeksforgeeks.org/ml-introduction-to-kernel-pca/
  Ref LDA: https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2
'''

# Import libraries -----------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

# Base Scale Data
variables = list(df.columns[:-1])
x = df.iloc[: ,0:-1].values
y = df.iloc[:, -1].values
x = StandardScaler().fit_transform(x)

# Convert Scaled data back to df
x = pd.DataFrame(x)


# Implement PCA --------------------------------------------
pca = PCA()
x_pca = pca.fit_transform(x)
x_pca = pd.DataFrame(x)

# Variance Explained by Principal Components ---------------
explained_variance = pca.explained_variance_ratio_
plt.bar(height=explained_variance, x= [0,1,2,3])
plt.show()


# Add Target & Columns Back to DataFrame -------------------
x_pca['target'] = y
x_pca.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'Target']
print(x_pca.head())

# Plot First 2 Principal Components ------------------------
'''
sns.pairplot(x_vars=["PC1"], y_vars=["PC2"], data=x_pca, hue='Target')
plt.show()
'''


### KERNEL PCA ---------------------------------------------
''' Kernel PCA maps data to a higher dimension and then 
    applies PCA.  Data that is not linier may be separable
    in higher dimensions. 
'''

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA

X, y = make_moons(n_samples = 500, noise = 0.02, random_state = 417) 
plt.scatter(X[:, 0], X[:, 1], c = y) 
plt.show() 

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

plt.title("PCA") 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c = y) 
plt.xlabel("Component 1") 
plt.ylabel("Component 2") 
plt.show() 
plt.close()


kpca = KernelPCA(kernel='rbf', gamma=17)
X_kpca = kpca.fit_transform(X) 
  
plt.title("Kernel PCA") 
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c = y) 
plt.show() 
plt.clf()




## Linear Discriminant Analysis --------------------------
'''
  LDA:  Tries to transform/project the data into a lower 
        dimensional space while preserving as much separability
        of the target variable as possible. 
        
'''
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Import Data 
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Categorical.from_codes(wine.target, wine.target_names)

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)
lda.explained_variance_ratio_

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
plt.show()


# Train Model using LDA Output

X_train, X_test, y_train, y_test = train_test_split(X_lda, y, random_state=1)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
confusion_matrix(y_test, y_pred)












