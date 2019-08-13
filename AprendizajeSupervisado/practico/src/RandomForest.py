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

#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel
#clf = ExtraTreesClassifier(n_estimators=45)
#clf = clf.fit(X.drop("PID",axis=1), y)
#model = SelectFromModel(clf, prefit=True)
#X_new = model.transform(X.drop("PID",axis=1))
#print(X_new.shape)              
#
## Create the model and evaluate it
#from sklearn.model_selection import train_test_split
#X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, test_size=0.3, random_state=619)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC

#params = {'criterion':('gini','entropy'), 
#          #'min_samples_leaf':min_samples_leaf,
#          'min_samples_split':min_samples_split,
#          'max_features':('auto','sqrt','log2',None),
#          'n_estimators' : n_estimators,
#          'min_impurity_decrease': min_impurity_decrease}
#
#tree = RFC(random_state=617,min_samples_leaf=2)
#tree_clf = GridSearchCV(tree, params, scoring='accuracy', cv=5, iid=True,n_jobs=56)
#tree_clf.fit(X_train, y_train)
#print('Best Decision Tree accuracy: ', tree_clf.best_score_)
#print('Best Decision Tree accuracy: ', tree_clf.score(X_valid,y_valid))
#print(grid.best_estimator_)


params={#'criterion':('gini','entropy'), 
        'min_samples_leaf':(1,2,3,4),
        'min_samples_split':(20,30,40),
        #'max_features':("auto","sqrt","log2",None),
       'n_estimators' : (95,100,105,110)}

clf = RFC(random_state=617,criterion='gini',max_features=None)#,class_weight="balanced")
tree_clf = GridSearchCV(clf, params, scoring='accuracy', cv=5, iid=True,n_jobs=56)
tree_clf.fit(X.drop("PID",axis=1), y)
print('Best Decision Tree accuracy: ',tree_clf.best_score_)
print(tree_clf.best_estimator_)


#X_new = model.transform(XX.drop("PID",axis=1))
X_new = XX.drop("PID",axis=1)
yy = tree_clf.predict(X_new)
yy = yy.astype(np.int)
submission = pd.DataFrame(list(zip(XX.PID, yy)), columns=["PID", "AdoptionSpeed"])
submission.to_csv("data/submission.csv", header=True, index=False)
