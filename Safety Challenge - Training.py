#!/usr/bin/env python
# coding: utf-8

# <h1>Safety Challenge - Training</h1>

# This is the training step of my submission for the [Grab AI for SEA - Safety Challenge](https://www.aiforsea.com/safety). Given a training feature file produced by the preprocessing step, this will produce a model file that will be used in the testing step.

# In[1]:


INPUT_FEATURE = 'dataset-ready.csv'
OUTPUT_MODEL = 'xgb.model'


# <h3>Import Libraries</h3>

# In[2]:


import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# <h3>Load Data</h3>

# In[3]:


df = pd.read_csv(INPUT_FEATURE)
df.head()


# <h3>Model Selection</h3>

# In[4]:


X = df.drop(['bookingID', 'label'], axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Train with default parameter as a baseline.

# In[5]:


model = XGBClassifier()
fit = model.fit(X_train, y_train)
print('training accuracy: ', fit.score(X_train, y_train))
print('testing accuracy: ', fit.score(X_test, y_test))
print('training ROC-AUC score: ', roc_auc_score(y_train, fit.predict_proba(X_train)[:,1]))
print('testing ROC-AUC score: ', roc_auc_score(y_test, fit.predict_proba(X_test)[:,1]))


# Tune the parameters.

# In[6]:


param_grid = {
    'max_depth': range(3, 10, 2),
    'min_child_weight':range(1, 6, 2),
#     'gamma':[i / 10.0 for i in range(0, 5)],
#     'subsample':[i / 10.0 for i in range(6, 10)],
#     'colsample_bytree':[i / 10.0 for i in range(6, 10)],
}
grid = GridSearchCV(estimator = XGBClassifier(), param_grid = param_grid, scoring='roc_auc', n_jobs=-1, cv=5)
grid.fit(X_train, y_train)
print('best parameter: ', grid.best_params_)
print('best score: ', grid.best_score_)
print('ROC-AUC score: ', roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]))


# It seems the tuning can't improve the accuracy significantly.

# Save the model for testing use.

# In[7]:


model_file = open(OUTPUT_MODEL,'wb')
pickle.dump(fit, model_file)
model_file.close()

