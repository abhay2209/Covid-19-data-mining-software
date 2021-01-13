#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('curl https://raw.githubusercontent.com/aamritpa/CMPT-Data-Covid-19/master/finalDataPart1.csv --output ./finalDataPart1.csv')
#get_ipython().system('curl https://raw.githubusercontent.com/aamritpa/CMPT-Data-Covid-19/master/finalDataPart2.csv --output ./finalDataPart2.csv')


# In[10]:


# Importing files
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
import datetime
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report


# In[11]:


def to_timestamp(d):
    return d.timestamp()


def makeDatetime(data):
    return datetime.datetime.strptime(data, '%Y-%m-%d')
    
def DataImportAndProcessing():
    # Importing Data
    data1 = pd.read_csv('finalDataPart1.csv')
    data2 = pd.read_csv('finalDataPart2.csv')
    # Appending IF NEEDED
    data = data1.append(data2)
    data=data.reset_index(drop=True)
    data = data.drop(columns=['index'])
    data['date_confirmation'] = data['date_confirmation'].apply(makeDatetime)
    data['date_confirmation'] = data['date_confirmation'].apply(to_timestamp) 
    # Encoding Label i.e outcome column
    data['outcome'].replace({'nonhospitalized':0, 'recovered':1, 'hospitalized':2, 'deceased':3},inplace=True)
    # Introducing dummies into data to get rid of all the categorical data
    encodedCountry = pd.get_dummies(data['country'])
    encodedProvince = pd.get_dummies(data['province'])
    encodedSex = pd.get_dummies(data['sex'])
    data = pd.concat([data, encodedSex, encodedProvince,encodedCountry], axis=1)
    print("length of filled data: ", len(data))
    data = data.drop(columns=['province','country','source', 'additional_information', 'sex'])
    return data

scoring = {"recall":make_scorer(recall_score, average = 'macro'),
           "recall_Deceased":  make_scorer(recall_score, labels = [3],  average= None),
           "Accuracy": make_scorer(accuracy_score)}


def GridXGB(X, y, mcw):
    grid = {"max_depth": [10, 20, 30]}
    
    clf = GridSearchCV(XGBClassifier(min_child_weight = mcw),
                  param_grid=grid,
                  scoring=scoring, refit=False, return_train_score=True, cv = 5, verbose = 2, n_jobs=-1)
    
    clf.fit(X, y)
    results = clf.cv_results_
    pd.DataFrame(results).to_csv(f"XGB_{mcw}.csv")
    display(pd.DataFrame(results))


def GridRF(X, y, md):
    grid = {"n_estimators": [50, 75, 100, 150, 200]}
    
    clf = GridSearchCV(RandomForestClassifier(max_depth = md),
                  param_grid=grid,
                  scoring=scoring, refit=False, return_train_score=True, cv =5, verbose = 2, n_jobs=-1)
    
    clf.fit(X, y)

    results = clf.cv_results_
    pd.DataFrame(results).to_csv(f"RF_{md}.csv")
    display(pd.DataFrame(results))



def GridKNN(X, y, w):
    grid = {"n_neighbors": [2, 3, 5, 7, 9, 11, 13, 20]}
    clf = GridSearchCV(KNeighborsClassifier(weights = w ),
                  param_grid=grid,
                  scoring=scoring, refit=False, return_train_score=True, cv = 5, verbose = 2, n_jobs=-1)

    clf.fit(X, y)
    results = clf.cv_results_
    pd.DataFrame(results).to_csv(f"KNN_{w}.csv")
    display(pd.DataFrame(results))



def XGB(X_train, X_valid, y_train, y_valid):
    Clf = make_pipeline(
      MinMaxScaler(),
      XGBClassifier(max_depth = 30, min_child_weight = 3)
    )
    Clf.fit(X_train, y_train)
    trainScore = Clf.score(X_train, y_train) 
    validationScore = Clf.score(X_valid, y_valid)
    print(f"Train Accuracy       : {trainScore * 100:.2f}%")
    print(f"Test Accuracy        : {validationScore * 100:.2f}%")
    predictY = Clf.predict(X_valid) 
    display(pd.DataFrame(classification_report(predictY, y_valid, output_dict=True)))

def RF(X_train, X_valid, y_train, y_valid):
    Clf = make_pipeline(
      MinMaxScaler(),
      RandomForestClassifier(n_estimators=100, max_depth=50)
    )
    Clf.fit(X_train, y_train)
    trainScore = Clf.score(X_train, y_train) 
    validationScore = Clf.score(X_valid, y_valid)
    predictY = Clf.predict(X_valid) 
    print(f"Train Accuracy       : {trainScore * 100:.2f}%")
    print(f"Test Accuracy        : {validationScore * 100:.2f}%")
    display(pd.DataFrame(classification_report(predictY, y_valid, output_dict=True)))

def KNNClassifier(X_train, X_valid, y_train, y_valid):
    Clf = make_pipeline(
      MinMaxScaler(),
      KNeighborsClassifier(n_neighbors = 3 , weights = 'uniform' )
    )
    Clf.fit(X_train, y_train)
    trainScore = Clf.score(X_train, y_train) 
    validationScore = Clf.score(X_valid, y_valid)
    predictY = Clf.predict(X_valid) 
    print(f"Train Accuracy       : {trainScore * 100:.2f}%")
    print(f"Test Accuracy        : {validationScore * 100:.2f}%")
    display(pd.DataFrame(classification_report(predictY, y_valid, output_dict=True)))


# In[15]:


def XGBAutomator(X, y):
    mcw = 1
    GridXGB(X, y, mcw)
    
    mcw = 3
    GridXGB(X, y, mcw)
    
    mcw = 5
    GridXGB(X, y, mcw)
    

def RFAutomator(X, y):
    # RF
    mdList = [None, 15, 25, 35, 50]
    for md in mdList:
        GridRF(X, y, md)

def KNNAutomator(X, y):
    # GridKNN
    w = 'uniform'
    GridKNN(X,y,w)

    w = 'distance'
    GridKNN(X,y,w)


def main():
    data = DataImportAndProcessing()
    
    dataNot3 = data[data.outcome != 3].reset_index(drop=True)
    data3 = data[data.outcome == 3].reset_index(drop=True)
  
    X_dataNot3 = dataNot3.drop(columns=['outcome']).to_numpy()
    y_dataNot3 = dataNot3['outcome'].to_numpy()

    X_data3 = data3.drop(columns=['outcome']).to_numpy()
    y_data3 = data3['outcome'].to_numpy()

    X_train_N3, X_valid_N3, y_train_N3, y_valid_N3 = train_test_split(X_dataNot3, y_dataNot3, test_size=0.40)

    X_train_3, X_valid_3, y_train_3, y_valid_3 = train_test_split(X_data3, y_data3, test_size=0.1)
    
    X_train_3 = pd.DataFrame(X_train_3)
    X_train_N3 = pd.DataFrame(X_train_N3)
    X_train = X_train_3.append(X_train_N3)#.to_numpy()
   
    X_valid_3 = pd.DataFrame(X_valid_3)
    X_valid_N3 = pd.DataFrame(X_valid_N3)
    X_valid = X_valid_3.append(X_valid_N3)#.to_numpy()
    

    y_train_3 = pd.DataFrame(y_train_3)
    y_train_N3 = pd.DataFrame(y_train_N3)
    y_train = y_train_3.append(y_train_N3).values.ravel()
    

    y_valid_3 = pd.DataFrame(y_valid_3)
    y_valid_N3 = pd.DataFrame(y_valid_N3)
    y_valid = y_valid_3.append(y_valid_N3).values.ravel()

    ## grid Search
    gridsearch = input("do you want to test grid search? y/n")
    if gridsearch == 'y':
        RFAutomator(X_train, y_train)
        KNNAutomator(X_train, y_train)
        XGBAutomator(X_train, y_train)

    ## best models 

    RF(X_train, X_valid, y_train, y_valid)
    
    XGB(X_train, X_valid, y_train, y_valid)
    
    KNNClassifier(X_train, X_valid, y_train, y_valid)

    
main()


# In[ ]:




