### The Purpose of this script is to include notes from a tutorial on label encoding in scikit learn




# Import processing library from scikit learn
from sklearn import preprocessing
import pandas as pd


## ENCODE LABELS (Two Methods)______________________________________________

# Instantiate Encoder
encoder = preprocessing.LabelEncoder()

# Labels 
Labels = ['setosa', 'versicolor', 'virginica']

# Encode Labels
def label_encoder_fit(Labels, encoder):
    # Encode Labels
    encoder.fit(Labels)
    # Print encoded labels
    for i, items in enumerate(encoder.classes_):
        print(items, '=>', i)

def label_encoder_transform(Labels, encoder):
    # Encode Labels
    Labels_encoded = encoder.transform(Labels)
    # Print Encoded Labels
    return Labels_encoded
    


## ONE HOT ENCODING (Using Pandas)____________________________________________
# Tutorial:  https://www.youtube.com/watch?v=BlaNvgfrHDg

# One Hot Encode
'''Note that Name & Gender are our features and age is our target'''
df = pd.DataFrame({})
df['Names'] =   ['Chris', 'Steve', 'John', 'Jackie']
df['Gender'] =  ['Male', 'Male', 'Male', 'Female']
df['Age'] =     [10, 15, 20, 30]


## ONE HOT ENCODING - P2________________________________________________________
'''Turorial = https://www.youtube.com/watch?v=xqBCYGvj55s

Step1:  One hot encode your entire dataframe.  This will result in new columns created for
        each of your features.  Print the dataframe to see the structure.  Note that Age is our
        target and that has remained unchanged.  No additional values columns were needed as it
        is numerical. Label values however were OneHotEnoded. 

Step2:  Create your x & y variables. 
        Use the columns of your OneHotEncoded dataframe in order to index which columns are associated
        with your x (independent) and y (dependent) variables. 

'''

## Step1 - One Hot Encode DataFrame
df_encoded = pd.get_dummies(df)
#print(df_encoded.columns)

# Define X values (equals independent variables)
x = df_encoded.ix[:, 'Names_Chris':'Gender_Male']

# Define Y values (dependent variables)
y = df_encoded.ix[:, 'Age']


### TRAIN ALGORITHM___________________________________________________________

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)


# Instanciate Algorithm
logreg = LogisticRegression()

# Fit Model
logreg.fit(X_train, y_train)

print('Logistic Regression score on the test set: {}'.format(logreg.score(X_test, y_test)))






























