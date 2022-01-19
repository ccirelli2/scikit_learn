'''
1.) Data Standardization, regularization, normalization
http://dataric.blogspot.com/2016/10/regularization-standardization-and.html

2.) Data standardization using scikit learn
https://scikit-learn.org/stable/modules/preprocessing.html

Numpy axis = https://www.sharpsightlabs.com/blog/numpy-axes-explained/

'''



# Test
X_test = np.array([ [1, -1 ,2],
                    [2, 0, 0],
                    [0, 1, -1]]
                    )

# Not-Standard Data
def get_mean_data(X_test):
    for n in range(0, len(X_test[:, 1])):
        feature = X_test[:, n]
        feature_mu = np.mean(feature)
        print(feature_mu)

def get_mean_standardized_data(X_test):
    x_scaled = preprocessing.scale(X_test)
    print(x_scaled.mean(axis=0))

