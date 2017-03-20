'''
Created on Mar 20, 2017

@author: abhijit.tomar
'''
import pandas as pd
import numpy as np
from sklearn import cross_validation
from constants import feat_list
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

if __name__ == '__main__':
    
    np.random.seed(1237) # seed to shuffle the train set

    n_folds = 3
    
    shuffle = False

    train = pd.read_csv("../../resources/data/train/train.csv",index_col=0) # the train dataset is now a Pandas DataFrame
    test = pd.read_csv("../../resources/data/test/test.csv",index_col=0) # the test dataset is now a Pandas DataFrame
    print(train.shape)
    print(test.shape)

    # Replace -999999 in var3 column with most common value 2 
    training = train.replace(-999999,2)

    X = training.iloc[:,:-1]
    y = training.TARGET

    features = feat_list.features
    print (features)
    X_sel = X[features]
    sel_test = test[features]
    X, y, X_submission = np.array(X_sel), np.array(y.astype(int)).ravel(), np.array(sel_test)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    ratio = float(np.sum(y == 1)) / np.sum(y==0)
    clfs = [RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced'),
            GradientBoostingClassifier(n_estimators=200,max_depth=5),
            AdaBoostClassifier(n_estimators=200),
            xgb.XGBClassifier(missing=9999999999,
                    max_depth = 6,
                    n_estimators=200,
                    learning_rate=0.1, 
                    nthread=4,
                    subsample=1.0,
                    colsample_bytree=0.5,
                    min_child_weight = 3,
                    scale_pos_weight = ratio,
                    reg_alpha=0.01,
                    seed=1301)]

    print ("Creating train and test sets for blending.")
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    skf = cross_validation.StratifiedKFold(y, n_folds, shuffle=True)
    
    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], n_folds))
        for i, (train, testidx) in enumerate(skf):
            print ("Fold", i+1)
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[testidx], y[testidx]
            if j < len(clfs)-1:
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
                    eval_set=[(X_test, y_test)])
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[testidx, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(axis=1)

    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print ("Saving Results.")
    submission = pd.DataFrame({"ID":test.index, "TARGET":y_submission})
    submission.to_csv("submission_RF_GBT_ABC_XGB.csv", index=False)