'''
Created on Nov 22, 2016

@author: abhijit.tomar
'''
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale
import json

def gen_chi_sq_feats(in_df,y,save_path):

    # First select features based on chi2 and f_classif
    p = 3
    
    X_bin = Binarizer().fit_transform(scale(in_df))
    selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
    selectF_classif = SelectPercentile(f_classif, percentile=p).fit(in_df, y)
    
    chi2_selected = selectChi2.get_support()
    chi2_selected_features = [ f for i,f in enumerate(in_df.columns) if chi2_selected[i]]
    print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
       chi2_selected_features))
    f_classif_selected = selectF_classif.get_support()
    f_classif_selected_features = [ f for i,f in enumerate(in_df.columns) if f_classif_selected[i]]
    print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
       f_classif_selected_features))
    selected = chi2_selected & f_classif_selected
    print('Chi2 & F_classif selected {} features'.format(selected.sum()))
    features = [ f for f,s in zip(in_df.columns, selected) if s]
    json.dump(features,open(save_path,'w'))