### THE PURPOSE OF THIS SCRIPT IS TO SHOW EXAMPLES OF SETTING UP ALGORITHMS WITH SCIKIT-LEARN

'''DOCUMENTATION

Source:         Video = Machine Learning Scikit-Learn by Cristi Vlad
                https://www.youtube.com/watch?v=mHEC8tB9ZCc&list=PLonlF40eS6nynU5ayxghbz2QpDsUAyCVF

cancer.DESCR:   Apparently Scikit learn data sets have pre-built attributes that you can call.

Datasets:       Author says that all data sets need to be in the form of numpy arrays in order
                to feed them into the machine learning model.

KNN             By default it uses 5 k points.


mglearn         Author uses this package.  Could not pip install.  Allows you to show the
                KNN clusters.

'''



### TITLE:  MACHINE LEARNING W/ SCIKIT-LEARN - THE CANCER DATASET - 1________________


## IMPORT LIBRARIES-------------------------------------------------------------------

# Import Scikit Learn Packages
from sklearn.datasets import load_breast_cancer

# Nearest Neighbor Algorithm
from sklearn.neighbors import KNeighborsClassifier

# Import Tool To Split Dataset into Train & Test Groups
from sklearn.model_selection import train_test_split

# Import Matplotlib
import matplotlib.pyplot as plt


## DATASET----------------------------------------------------------------------------

# Load Dataset
cancer = load_breast_cancer()

# Print Features of Dataset
'''
print(cancer.DESCR)
print(cancer.feature_names)
print(cancer.target_names)
print(cancer.data)
print(cancer.shape)
'''

## KNN MACHINE LEARNING ALGORITHMS----------------------------------------------------


# Split dataset into training and test sets
'''
Stratify:       What does this mean?
'''
def generate_prediction_KNN(cancer):
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify = cancer.target,
                                                    random_state = 42)
    # Instantiate KNN Algorithm
    knn = KNeighborsClassifier()
    # Fit to training data
    knn.fit(x_train, y_train)
    # Print Results
    '''
    knn.score:      Returns the accuracy on the training set
    '''
    print('Accuracy of KNN n-5, on training set:{}'.format(knn.score(x_train, y_train)))
    print('Accuracy of KNN n-5, on the test set:{}'.format(knn.score(x_test, y_test)))


# TUNING HYPER PARAMETERS
'''
random_state:       use 66
number neighbors:   run a for loop to see which works best, 1-10.
'''

range_neighbors = range(1,10)

def generate_prediction_KNN_range_neighbors(dataset, range_neighbors):

    # Step1:  Split dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(
                                    dataset.data, # features
                                    dataset.target,
                                    stratify = dataset.target,
                                    random_state = 42)
    training_accuracy_list = []
    test_accuracy_list = []

    for num in range_neighbors:
        clf = KNeighborsClassifier(n_neighbors = num)
        clf.fit(x_train, y_train)
        training_accuracy_list.append(clf.score(x_train, y_train))
        test_accuracy_list.append(clf.score(x_test, y_test))

    plt.plot(range_neighbors, training_accuracy_list, label = 'Accuracy of training')
    plt.plot(range_neighbors, test_accuracy_list, label = 'Accuracy of test')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.title('Cancer Data Set KNN Algorithm')
    plt.legend()
    plt.show()



generate_prediction_KNN_range_neighbors(cancer, range_neighbors)
