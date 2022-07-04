"""
Examples of how to build an ml pipeline using scikit

# Pipeline
- Pass a series of key value pairs to the Pipeline constructor whereby the string is the name of the transformation
    and the key is the function or estimator object.

References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html
    - https://towardsdatascience.com/building-a-machine-learning-pipeline-3bba20c2352b
    - https://github.com/ezgigm/Project3_TanzanianWaterWell_Status_Prediction/blob/master/STEP2_Modeling.ipynb
    - http://rasbt.github.io/mlxtend/
"""

import os
import pandas as pd
import category_encoders as ce
from decouple import config as d_config
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt
import seaborn as sns

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_DATA = d_config("DIR_DATA")

# Package Settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Load Data
df_car = pd.read_csv(os.path.join(DIR_DATA, "car_insurance_claims_joined.csv"))
df_car = df_car.fillna(0)

#########################################################################################################################
# Create Dataset
#########################################################################################################################

# Create a Pre-Processing Pipeline
cat_columns = ["MSTATUS", "GENDER", "EDUCATION", "OCCUPATION", "CAR_USE", "CAR_TYPE", "RED_CAR", "REVOKED",
               "URBANICITY", "PARENT1"]
num_columns = ["AGE", "HOMEKIDS", "INCOME_0", "HOME_VAL_0", "TRAVTIME", "BLUEBOOK_0", "TIF", "MVR_PTS",
               "CAR_AGE"]
target_column = ["CLAIM_FLAG"]
columns_to_drop = ["ID", "CLM_FREQ", "CLM_AMT_0", "CLAIM_FLAG", "CLAIM_FLAG_CAT"]

# Split X & Y
y = df_car[target_column].values.ravel()
X = df_car.drop(columns_to_drop, axis=1)

# Create Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

########################################################################################################################
# Make Pipeline
########################################################################################################################

# Instantiate Transformers
scaler = RobustScaler()
encoder = ce.TargetEncoder(cols=cat_columns)

# Add Transformers to Pipeline
num_transformer = make_pipeline(scaler)
cat_transformer = make_pipeline(encoder)

# Create Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_columns),
        ("cat", cat_transformer, cat_columns)
    ])

# Model
model_lr = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=123, max_iter=10_000)

# Create Pipe
pipe = make_pipeline(preprocessor, model_lr)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

# Call Score to get
score = pipe.score(X_test, y_test)
print(f"Logistic Regression Accuracy Score => {score}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=pipe.classes_)
display.plot()
plt.show()