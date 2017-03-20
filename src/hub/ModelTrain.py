'''
Created on Mar 20, 2017

@author: abhijit.tomar
'''
from preproc import ReadIn
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
if __name__ == '__main__':
    
    train , test = ReadIn.get_cleaned()
    test_id = test.ID
    test = test.drop(["ID"],axis=1)
    X_train, X_test, y_train, y_test = ReadIn.get_TT_split(train, test, 0.30)
    model_names=[
                "ExtraTreesClassifier",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "AdaBoostClassifier"
                ]
    clf_list = [ExtraTreesClassifier(random_state=1729),
                RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced'),
                GradientBoostingClassifier(n_estimators=200,max_depth=5),
                AdaBoostClassifier(n_estimators=200)]
    
    for idx, clf in enumerate(clf_list):
        
        
        selector = clf.fit(X_train, y_train)
        
        # plot most important features
        feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)
        feat_imp[:40].plot(kind='bar', title='Feature Importances according to '+model_names[idx], figsize=(12, 8))
        plt.ylabel('Feature Importance Score')
        plt.subplots_adjust(bottom=0.3)
        plt.savefig('../../resources/plots/models/'+model_names[idx]+'.png')
        plt.show()
        
        # clf.feature_importances_ 
        fs = SelectFromModel(selector, prefit=True)
        
        X_train_np = fs.transform(X_train)
        X_test_np = fs.transform(X_test)
        test_np = fs.transform(test)
                
        ## # Train Model
        # classifier from xgboost
        m2_xgb = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth = 4, \
        seed=1729)
        m2_xgb.fit(X_train_np, y_train, eval_metric="auc", verbose = False,
                   eval_set=[(X_test_np, y_test)])
        
        # calculate the auc score
        print("Roc AUC: ", roc_auc_score(y_test, m2_xgb.predict_proba(X_test_np)[:,1],
                      average='macro'))
                      
        ## # Submission
        probs = m2_xgb.predict_proba(test_np)
        
        submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
        submission.to_csv(model_names[idx]+"-XGB.csv", index=False)