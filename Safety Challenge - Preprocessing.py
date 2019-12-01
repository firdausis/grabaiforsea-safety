#!/usr/bin/env python
# coding: utf-8

# <h1>Safety Challenge - Preprocessing</h1>

# This is the preprocessing step of my submission for the [Grab AI for SEA - Safety Challenge](https://www.aiforsea.com/safety). Given a dataset, this will produce a file containing feature data that will be used in the training or testing step.
# For training purpose, assume this [dataset](https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/safety.zip) is already extracted in the same folder with this notebook. For testing purpose, the config below may need to be changed.

# In[1]:


INPUT_DATASET_FEATURES_DIR = './safety/features/'
INPUT_DATASET_LABEL_DIR = './safety/labels/'
OUTPUT_FEATURES = 'dataset-ready.csv'


# <h3>Import Libraries</h3>

# In[2]:


import pandas as pd
import numpy as np
import glob, os


# <h3>Load Data</h3>

# In[3]:


telematics = pd.concat(map(pd.read_csv, glob.glob(os.path.join(INPUT_DATASET_FEATURES_DIR, "*.csv"))))
labels = pd.concat(map(pd.read_csv, glob.glob(os.path.join(INPUT_DATASET_LABEL_DIR, "*.csv"))))


# <h3>Exploration and Cleansing</h3>

# <h4>Telematics</h4>

# In[4]:


telematics.head()


# In[5]:


telematics.shape[0]


# In[6]:


telematics['bookingID'].unique().size


# Remove data with invalid speed (<0 and >300 km/s)

# In[7]:


filtered_telematics = telematics[(telematics['Speed'] >= 0) & (telematics['Speed'] <= 83)]
filtered_telematics.shape[0]


# Remove data with low accuracy

# In[8]:


filtered_telematics = filtered_telematics[filtered_telematics['Accuracy'] <= 50]
filtered_telematics.shape[0]


# Remove invalid trips

# In[9]:


trips = filtered_telematics[['bookingID','second']].groupby('bookingID').agg(['max','count'])
trips.head()


# In[10]:


bookingID_to_remove = trips[(trips[('second', 'max')] > 43200) | (trips[('second', 'count')] < 100)].index.tolist()
filtered_telematics = filtered_telematics[~filtered_telematics['bookingID'].isin(bookingID_to_remove)]
filtered_telematics.shape[0]


# In[11]:


filtered_telematics['bookingID'].unique().size


# <h4>Labels</h4>

# In[12]:


labels.head()


# In[13]:


filtered_labels = labels[labels['bookingID'].isin(filtered_telematics['bookingID'].unique())]
filtered_labels.shape[0]


# In[14]:


filtered_labels['bookingID'].unique().size


# Since there are some duplicate bookings, remove them and keep label 1 (dangerous) if any.

# In[15]:


filtered_labels = filtered_labels.sort_values(by='label', ascending=False)
filtered_labels = filtered_labels.drop_duplicates(subset='bookingID', keep='first')
filtered_labels.shape[0]


# <h3>Feature Extraction</h3>

# Calculate magnitude of acceleration and gyro

# In[16]:


filtered_telematics['acceleration'] = np.sqrt(filtered_telematics['acceleration_x']**2                                               + filtered_telematics['acceleration_y']**2                                               + filtered_telematics['acceleration_z']**2)
filtered_telematics['gyro'] = np.sqrt(                                      filtered_telematics['gyro_x']**2                                       + filtered_telematics['gyro_y']**2                                       + filtered_telematics['gyro_z']**2)
# filtered_telematics.head()


# Extract features:
# - Speed (max, mean, IQR, max change)
# - Acceleration (min, max, mean, IQR, max change)
# - Gyro (min, max, mean, IQR, max change)
# - Duration
# - Distance
# - Rotation

# In[17]:


def iqr():
    def iqr_(x):
        return x.quantile(0.75) - x.quantile(0.25)
    iqr_.__name__ = 'iqr'
    return iqr_

aggregated_telematics = filtered_telematics[['bookingID','Speed','acceleration','gyro','second']]    .groupby('bookingID')    .agg({'Speed': [np.max, np.mean, iqr()],          'acceleration': [np.max, np.mean, iqr()],          'gyro': [np.max, np.mean, iqr()],
         'second': [np.max]})
# aggregated_telematics.head()


# In[18]:


df = pd.DataFrame()
df['speed_max'] = aggregated_telematics[('Speed','amax')]
df['speed_mean'] = aggregated_telematics[('Speed','mean')]
df['speed_iqr'] = aggregated_telematics[('Speed','iqr')]

df['acceleration_max'] = aggregated_telematics[('acceleration','amax')]
df['acceleration_mean'] = aggregated_telematics[('acceleration','mean')]
df['acceleration_iqr'] = aggregated_telematics[('acceleration','iqr')]

df['gyro_max'] = aggregated_telematics[('gyro','amax')]
df['gyro_mean'] = aggregated_telematics[('gyro','mean')]
df['gyro_iqr'] = aggregated_telematics[('gyro','iqr')]

df['duration'] = aggregated_telematics[('second','amax')]
df['distance'] = df['duration'] * df['speed_mean']
df['rotation'] = df['duration'] * df['gyro_mean']


# In[19]:


sorted_labels = filtered_labels.sort_values(by='bookingID')
sorted_bookingIDs = sorted_labels['bookingID']
filtered_telematics = filtered_telematics.sort_values(by=['bookingID','second'])
df['speed_max_change'] =     [filtered_telematics[filtered_telematics['bookingID'] == bid]['Speed'].diff().abs().max() for bid in sorted_bookingIDs]
df['acceleration_max_change'] =     [filtered_telematics[filtered_telematics['bookingID'] == bid]['acceleration'].diff().abs().max() for bid in sorted_bookingIDs]
df['gyro_max_change'] =     [filtered_telematics[filtered_telematics['bookingID'] == bid]['gyro'].diff().abs().max() for bid in sorted_bookingIDs]


# In[20]:


df['label'] = sorted_labels['label'].tolist()


# In[21]:


df.head()


# In[22]:


df.info()


# In[23]:


df.to_csv(OUTPUT_FEATURES)

