#!/usr/bin/env python
# coding: utf-8

# # Diplodatos Kaggle Competition

# We present this peace of code to create the baseline for the competition, and as an example of how to deal with these kind of problems. The main goals are that you:
# 
# 1. Learn
# 1. Try different models and see which one fits the best the given data
# 1. Get a higher score than the given one in the current baseline example
# 1. Try to get the highest score in the class :)

# In[1]:


# Import the required packages
import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# load the given labels
breed = pd.read_csv('data/breed_labels.csv')
color = pd.read_csv('data/color_labels.csv')
state = pd.read_csv('data/state_labels.csv')

# In[6]:

original_df = pd.read_csv('data/train.csv')

# In[10]:

def transform_data(train_data_fname, test_data_fname):
    def transform_columns(df):
        df = df.drop(["Description"], axis=1)
        df.Type = df.Type.replace({1: 'Dog', 2: 'Cat'})
        df.Gender = df.Gender.replace({1:'Male', 2:'Female', 3:'Mixed'})
        df.MaturitySize = df.MaturitySize.replace({1:'S', 2:'M', 3:'L', 4:'XL', 0:'N/A'})
        df.FurLength = df.FurLength.replace({1:'S', 2:'M', 3:'L', 0:'N/A'})
        df.Vaccinated = df.Vaccinated.replace({1:'T', 2:'N', 3:'N/A'})
        df.Dewormed = df.Dewormed.replace({1:'T', 2:'F', 3:'N/A'})
        df.Sterilized = df.Sterilized.replace({1:'T', 2:'F', 3:'N/A'})
        df.Health = df.Health.replace({1:'Healthy', 2: 'MinorInjury', 3:'SeriousInjury', 0: 'N/A'})
        df.Color1 = df.Color1.replace(dict(list(zip(color.ColorID, color.ColorName)) + [(0, "N/A")]))
        df.Color2 = df.Color2.replace(dict(list(zip(color.ColorID, color.ColorName)) + [(0, "N/A")]))
        df.Color3 = df.Color3.replace(dict(list(zip(color.ColorID, color.ColorName)) + [(0, "N/A")]))
        df.Breed1 = df.Breed1.replace(dict(list(zip(breed.BreedID, breed.BreedName)) + [(0, "N/A")]))
        df.Breed2 = df.Breed2.replace(dict(list(zip(breed.BreedID, breed.BreedName)) + [(0, "N/A")]))
        return df
    
    df_train = pd.read_csv(train_data_fname)
    df_train = transform_columns(df_train)
    df_test = pd.read_csv(test_data_fname)
    df_test = transform_columns(df_test)
    
    df = pd.concat([df_train, df_test], sort=True)

    # set dummy variables for everything
    # except from Age, Quantity, Fee
    df = pd.get_dummies(df)
    # get train and test back
    n = len(df_train)
    df_train = df.iloc[:n]
    df_test = df.iloc[n:]
    
    y = df_train['AdoptionSpeed']
    X = df_train.drop('AdoptionSpeed', axis=1)
    yy = None
    XX = df_test.drop('AdoptionSpeed', axis=1)

    return X, y, XX, yy


# Load the data...

# In[11]:


X, y, XX, yy = transform_data("data/train.csv", "data/test.csv")

# Utility function to report best scores
def report(results, n_top=3):
  for i in range(1, n_top + 1):
    candidates = np.flatnonzero(results['rank_test_score'] == i)
    for candidate in candidates:
      print("Model with rank: {0}".format(i))
      print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
      print("Parameters: {0}".format(results['params'][candidate]))
      print("")

# Create the model and evaluate it

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=619)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

results = pd.DataFrame(columns=('clf', 'best_acc'))


# In[ ]:

from sklearn.ensemble import RandomForestClassifier as RFC

#min_samples_leaf = np.arange(1,5,1)
min_samples_split = np.arange(50,70,5)
#max_features = np.arange(10,50,10)
n_estimators = np.arange(40,60,2)
min_impurity_decrease = np.logspace(-5,-5.5,3)


scores = np.zeros(100)
for i in range(100):
  print(i)
  tree = RFC(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features=None, max_leaf_nodes=None,
           min_impurity_decrease=1e-05, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=54,
           min_weight_fraction_leaf=0.0, n_estimators=50,
           n_jobs=56, oob_score=False, random_state=None,
           verbose=0, warm_start=False)

  tree.fit(X_train.drop("PID",axis=1), y_train)
  #print(tree.feature_importances_)
  #print('Best Decision Tree accuracy: ', tree.score(X_valid.drop("PID",axis=1),y_valid))
  scores[i] = tree.score(X_valid.drop("PID",axis=1),y_valid)

print(scores.mean())
print(scores.std())
