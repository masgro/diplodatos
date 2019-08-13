#!/usr/bin/env python

# Import the required packages
import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the given labels
breed = pd.read_csv('data/breed_labels.csv')
color = pd.read_csv('data/color_labels.csv')
state = pd.read_csv('data/state_labels.csv')

original_df = pd.read_csv('data/train.csv')

# Utility function to report best scores
def report(results, n_top=3):
  for i in range(1, n_top + 1):
    candidates = np.flatnonzero(results['rank_test_score'] == i)
    for candidate in candidates:
      print("Model with rank: {0}".format(i))
      print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
      print("Parameters: {0}".format(results['params'][candidate]))
      print("")

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
X, y, XX, yy = transform_data("data/train.csv", "data/test.csv")

# Create the model and evaluate it
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=619)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

results = pd.DataFrame(columns=('clf', 'best_acc'))

from lightgbm import LGBMClassifier as lgb


# Additional parameters:
early_stop = 500
verbose_eval = 100
num_rounds = 10000
n_splits = 5

clf = lgb(objective='multiclass',
          metric=None,
          max_depth=-1,
          verbosity=-1,
          random_state=42,
          boosting_type='dart',
          min_child_samples=20,
          n_estimators = 230,
          subsample_for_bin=10000)

params = {#'boosting_type': ('gbdt','dart','goss'),
          #'min_child_weight' : (1.e-6,1.e-5,1.e-4),
          #'min_child_samples': (15,20,25),
          'num_leaves': (55,60,65),
          'learning_rate': (0.03,0.04,0.05),
          'n_estimators':(220,230,240)
          #'subsample_for_bin':(9000,10000,11000)
          }

model = GridSearchCV(clf, params, scoring='accuracy', cv=5, iid=True,n_jobs=56)
model.fit(X_train.drop("PID",axis=1), y_train)
print('Best Decision Tree accuracy: ', model.best_score_)
print('Best Decision Tree accuracy: ', model.score(X_valid.drop("PID",axis=1),y_valid))
print(model.best_estimator_)

yy = model.predict(XX.drop("PID",axis=1))
yy = yy.astype(np.int)
submission = pd.DataFrame(list(zip(XX.PID, yy)), columns=["PID", "AdoptionSpeed"])
submission.to_csv("data/submission.csv", header=True, index=False)
