'''
Created on Mar 20, 2017

@author: abhijit.tomar
'''
import pandas as pd
import numpy as np
def get_cleaned():
    train = pd.read_csv("../../resources/data/train/train.csv") # the train dataset is now a Pandas DataFrame
    test = pd.read_csv("../../resources/data/test/test.csv") # the test dataset is now a Pandas DataFrame
    # remove constant columns (std = 0)
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    
    # remove duplicated columns
    remove = []
    cols = train.columns
    for i in range(len(cols)-1):
        v = train[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,train[cols[j]].values):
                remove.append(cols[j])
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    
    return train,test

from sklearn.cross_validation import train_test_split
def get_TT_split(train,test,split_ratio):
    
    # split data into train and test
    
    X = train.drop(["TARGET","ID"],axis=1)
    y = train.TARGET.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=1729)
    
    return X_train, X_test, y_train, y_test