'''
Created on Nov 29, 2016

@author: abhijit.tomar
'''
import pandas as pd
import json
def gen_correlations(train,load_path,save_path):
    
    features = json.load(open(load_path,'r'))
    # Make a dataframe with the selected features and the target variable
    X_sel = train[features+['TARGET']]
    cor_mat = X_sel.corr()
    
    threshold = 0.7
    
    important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0]) \
        .unstack().dropna().to_dict()
    unique_important_corrs = pd.DataFrame(
        list(set([(tuple(sorted(key)), important_corrs[key]) \
        for key in important_corrs])), columns=['attribute pair', 'correlation'])
    # sorted by absolute value
    unique_important_corrs = unique_important_corrs.ix[
        abs(unique_important_corrs['correlation']).argsort()[::-1]]
    pd.DataFrame.to_csv(unique_important_corrs, save_path)