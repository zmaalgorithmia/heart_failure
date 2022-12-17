#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# # Exploratory Data Analysis

# In[2]:


# Read the dataset
df = pd.read_csv('./dataset/heart_failure.csv')
df.head()


# In[3]:


# Review the summary statistics of the training dataset
df.describe().T


# # Partitioning

# In[4]:


# Partition the dataset into training and testing datasets
X, y = df.drop('death_event', axis=1), df['death_event']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Feature Transformation

# In[5]:


# Check whether there are non-numeric features in the training dataset
non_numeric_features = X_train.select_dtypes(exclude='number').columns.tolist()
if len(non_numeric_features) == 0:
    print("There are no non-numeric features in the dataset.")


# In[6]:


# Create a column transformer to impute missing values and normalize the data
col_transformation_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

all_columns = X_train.columns.tolist()

columns_transformer = ColumnTransformer(transformers=[
    ('cols', col_transformation_pipeline, all_columns),
])


# # Model Training & Evaluation

# In[7]:


''' 
Define the steps in the training pipeline: 
1) preprocessing with the column transformer
2) fit the trainig data with the random forest classifier
'''
rf_classifier = RandomForestClassifier(
    n_estimators=11, criterion='entropy', random_state=0)

rf_model_pipeline = Pipeline(steps=[
    ('preprocessing', columns_transformer),
    ('rf_model', rf_classifier),
])

rf_model_pipeline.fit(X_train, y_train)


# In[8]:


# Score the testing data using the random forrest model
y_pred = rf_model_pipeline.predict(X_test)


# In[9]:


# Calculae the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


# In[10]:


print(classification_report(y_test, y_pred))


# # Hyperparameter Tuning

# In[11]:


'''
Tune hyperparameters using grid search:
1) n_estimators: The number of trees in random forest,
2) criterion: The function to measure the quality of a split,
3) max_depth : The maximum depth of the tree.
'''
params_dict = {
    'rf_model__n_estimators': np.arange(5, 10, 1),
    'rf_model__criterion': ['gini', 'entropy'],
    'rf_model__max_depth': np.arange(10, 20, 5)
}

random_forest_model = GridSearchCV(
    rf_model_pipeline, params_dict, cv=10, n_jobs=-1)
random_forest_model.fit(X_train, y_train)


# In[12]:


# Score the testing data using the tuned model
y_pred_new = random_forest_model.predict(X_test)


# In[13]:


# Display confugison matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_new)


# In[14]:


# Print out the classification report
print(classification_report(y_test, y_pred_new))


# # Serialization

# In[15]:


# Serialize and save the trained model
joblib.dump(rf_model_pipeline, "random_forest_model.pkl")


# # Testing the Serialized Model

# In[16]:


model = joblib.load("random_forest_model.pkl")


# In[17]:


predictions = model.predict_proba(X_test)


# In[18]:


predictions
