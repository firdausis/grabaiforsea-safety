#!/usr/bin/env python
# coding: utf-8

# <h1>Safety Challenge - Testing</h1>

# This is the testing step of my submission for the [Grab AI for SEA - Safety Challenge](https://www.aiforsea.com/safety). Given a testing feature file produced by the preprocessing step and a model produced by the training step, this will measure the model performance.

# In[1]:


INPUT_FEATURE = 'dataset-ready.csv'
INPUT_MODEL = 'xgb.model'


# <h3>Import Libraries</h3>

# In[2]:


import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# <h3>Load Data</h3>

# In[3]:


df = pd.read_csv(INPUT_FEATURE)
df.head()


# <h3>Load Model</h3>

# In[4]:


model_file = open(INPUT_MODEL,'rb')
model = pickle.load(model_file)
model_file.close()


# <h3>Evaluation</h3>

# In[5]:


X = df.drop(['bookingID', 'label'], axis=1)
y = df['label']
print('accuracy: ', model.score(X, y))
print('ROC-AUC score: ', roc_auc_score(y, model.predict_proba(X)[:,1]))

