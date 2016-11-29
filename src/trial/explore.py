'''
Created on Nov 18, 2016
022 25670319
@author: abhijit.tomar
'''
import pandas as pd
import numpy as np

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import re
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)   
from trial import chi_sq_features,feature_correlations

def gen_plots(in_df,target_col,reg_pat,save_dir,x_label,density_gen=False):
    
    cols = in_df.columns
    req_cols = []
    req_mx_vals = []
    #find out which columns indicate a quantity of something. also find max value of this column
    for col in cols:
        if re.match(reg_pat,col):
            req_cols.append(col)
            req_mx_vals.append(in_df[col].max())
    
    zip_req = zip(req_cols,req_mx_vals)
    #for a quantity column, plot it against number of customers
    
    for col,mx_val in zip_req:
        print(col,mx_val)
        pl_size=10
        if mx_val<pl_size and mx_val>0:
            pl_size=mx_val
            
        sns.FacetGrid(in_df, hue=target_col, size=pl_size) \
       .map(plt.hist, col) \
       .add_legend()
       
        plt.xlabel(x_label+col)
        plt.ylabel('Number of customers in train')
        plt.savefig(save_dir+col+'.png')
        plt.close()      
        
        if density_gen:
            try:
                sns.FacetGrid(in_df, hue=target_col, size=pl_size) \
               .map(sns.kdeplot, col) \
               .add_legend()
                plt.xlabel('Density '+col)
                plt.savefig(save_dir+col+'_density.png')
                plt.close()
            except np.linalg.linalg.LinAlgError as e:
                print('Too may zeros in '+col)

def gen_log_plots(in_df,target_col,save_dir,x_label):                 
    
    cols = in_df.columns
    
    #find out which columns indicate a quantity of something. also find max value of this column
    for col in cols:
        in_df[col]=in_df[col].dropna()
        in_df[col].replace(0, np.nan).dropna().map(np.log).hist(bins=1000)
        plt.xlabel(x_label+col)
        plt.ylabel('Number of customers in train')
        plt.savefig(save_dir+col+'.png')
        plt.close()
        
if __name__=='__main__':
    train = pd.read_csv("../../resources/data/train/train.csv") # the train dataset is now a Pandas DataFrame
    test = pd.read_csv("../../resources/data/test/test.csv") # the test dataset is now a Pandas DataFrame
    gen_plots(train,'TARGET','^num_var[0-9]{1,2}$','../../resources/plots/num_vars/','Number of ',True)
    '''
    num_var1 6
    num_var4 7
    num_var5 15
    num_var6 3
    num_var8 3
    num_var12 15
    num_var13 18
    num_var14 12
    num_var17 27
    num_var18 3
    num_var20 3
    num_var24 6
    num_var26 33
    num_var25 33
    num_var28 0
    num_var27 0
    num_var29 3
    num_var30 33
    num_var31 27
    num_var32 12
    num_var33 6
    num_var34 3
    num_var35 36
    num_var37 114
    num_var40 3
    num_var41 0
    num_var39 3
    num_var42 18
    num_var44 3
    num_var46 0
    '''
    gen_plots(train,'TARGET','^var[0-9]{1,2}$','../../resources/plots/vars/','Value of ',True)
    '''
    var3 238
    var15 105
    var36 99
    var21 30000
    var38 22034738.76
    '''
    '''
    Too may zeros in num_var6
    Too may zeros in num_var18
    Too may zeros in num_var20
    Too may zeros in num_var28
    Too may zeros in num_var27
    Too may zeros in num_var29
    Too may zeros in num_var33
    Too may zeros in num_var34
    Too may zeros in num_var41
    Too may zeros in num_var46
    '''
    # Add feature that counts the number of zeros in a row
    X = train.iloc[:,:-1]
    y = train.TARGET
    
    X['n0'] = (X==0).sum(axis=1)
    train['n0'] = X['n0']
    
    gen_log_plots(train,'TARGET','../../resources/plots/logs/','Log of ')
    
    chi_sq_features.gen_chi_sq_feats(X,y)
    
    feature_correlations.gen_correlations(train, y)