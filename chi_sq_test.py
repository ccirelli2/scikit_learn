'''
    Definition:
    Url: https://machinelearningmastery.com/chi-squared-test-for-machine-learning/

    Video; https://www.youtube.com/watch?time_continue=5&v=IrZOKSGShC8
    https://en.wikipedia.org/wiki/Chi-squared_test
'''

# Load Libraries ---------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Load Dataset -----------------------------------------------
wind_load = datasets.load_wine()
X, y = datasets.load_wine(return_X_y=True)
feature_names = wind_load['feature_names']


# Create Contingency Table -----------------------------------
'''Purpose: Summarize the categorical variable as it pertaines
    to another categorical variable and or the target. 
    Here we will use the target and frequency. 
    
    The table was called a contingency table by Karl Pearson with the
    intent to help determine whether one varaible is contingent or
    depends on another. 
'''

def get_df_or_cont_table(X, y, feature_names, cont=True):
    # Create DataFrame
    df = pd.DataFrame({})
    for n in range(0, len(X[1,:])):
        col_data = X[:, n]
        col_name = feature_names[n]
        df[col_name] = col_data
    df['target'] = y
    # Return Value
    if cont==True:
        return df.groupby(['target']).count()
    else:
        return df


# Dataframe & Contingency Tables
df = get_df_or_cont_table(X, y, feature_names, cont=False)
df_cont = get_df_or_cont_table(X, y, feature_names, cont=True)



# Chi-Squared Test ---------------------------------------------
''' Def: Is a statistical hypothesis test that assumes (the null-hyp) 
         the observed frequencies for a categorical variable match the 
         exp frequencies for the same variable.  The test calculates
         a statistic that has a chi-squared distribution. 

        The chi-squared test first calculate the expected frequencies for
        the group and then determines whether the division of the groups, 
        called observed frequencies, matches the expected frequencies. 

        Result: a test statistic that has a chi-squared distribution and 
        can be interpreted to reject or fail to reject the null hypothesis. 
        When the frequency is far from the expected we get a large result, and small
        when similar. The variables are considered dependent when the observed and 
        expected are different. 
'''

def plot_chsq_dist():
    chsq_dist = np.random.chisquare(10, size=10000)
    plt.hist(chsq_dist)
    plt.show()

# Degrees of Freedom
''' Calculated using the rows and columns of the frequency table
    (rows - 1) * (cols -1)
    If p-value <= alpha: significant result, reject null hypothesis (H0), dependent.
    If p-value > alpha: not significant result, fail to reject null hypothesis (H0), 
    independent
    
    Independent: if the observed and expected frequencies are similar
    Dependent: if the observed and expected frequencies are different. 
    '''

# Library
'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html'
from scipy.stats import chi2_contingency
from scipy.stats import chi2

def perform_chi_sqr_test(df_cont, feature_names):
    # Create DataFrame to house results
    df_results = dict()

    # Reset Index & Choose One Variable
    df_reset_index = df_cont.reset_index()
  
    # Iterate Over Target & Feature
    for feature in feature_names:
        df_single_var = df_reset_index[['target', feature]]
        # Run Chi-squared test function 
        stat, p, dof, expected = chi2_contingency(df_single_var)
        # Interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        if abs(stat) >= critical:
            df_results[feature] = 'Dependent'
        else: 
            df_results[feature] = 'Independent'

        # Interpret P Value
        alpha = 1.0 - prob
        print('P value => {}'.format(p))
        
        if p <= alpha:
            print('Dependent and reject H0')
        else: 
            print('Independent, failt to reject H0')
    
    # Return DataFrame of Results
    return df_results

df_chi_sq_results = perform_chi_sqr_test(df_cont, feature_names)
print(df_chi_sq_results)



