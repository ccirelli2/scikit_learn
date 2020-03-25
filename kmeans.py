''' Kmeans tutorial
    URL:  https://towardsdatascience.com/customer-segmentation-with-machine-learning-a0ac8c3d4d84

Content:
    Business Case
    Data Preparation
    Segmentation with K-means Clustering
    Hyperparameter Tuning
    Visualization and Interpretation of the Results

'''

# IMPORT LIBRARIES -------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# LOAD DATA --------------------------------------------------
dir_data = r'/home/cc2/Desktop/repositories/scikit_learn/data'
afile = r'Mall_Customers.csv'
df = pd.read_csv(dir_data + '/' + afile)
df.drop(['CustomerID'], axis=1, inplace=True)

# EXPLORATORY DATA ANALYSIS ----------------------------------

# Get Standard Measures of Data
def data_inspection(df):
    print(df.describe(), '\n')
    print('Columns => {}\n'.format(df.columns))
    print('Null values => \n{}'.format(df.isnull().sum()))

data_inspection(df)


# Plot Gender
def plot_gender(df):
    df_gender = df.groupby('Gender')['Gender'].count()
    df_gender.plot(kind='bar')
    plt.title('Gender Count')
    plt.show()

def plot_age(df):
    sns.axes_style('dark')
    sns.violinplot(y=df['Age'])
    plt.title('Age')
    plt.show()


def plot_income(df):
    bins = pd.cut(df['Annual Income (k$)'], [0, 25, 50, 75, 100, 125, 150]) 
    df_income = df.groupby(bins)['Annual Income (k$)'].agg('count')
    income_mu = np.mean(df['Annual Income (k$)'])
    df_income.plot.bar()
    plt.axvline(income_mu, color='r', linestyle='--')
    plt.title('Bar Plot')
    plt.show()


def plot_score(df):
    plt.scatter(df['Spending Score (1-100)'], df['Annual Income (k$)'])
    plt.show()



