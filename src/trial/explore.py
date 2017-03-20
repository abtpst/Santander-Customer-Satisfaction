'''
Created on Nov 18, 2016
022 25670319
@author: abhijit.tomar
'''
import pandas as pd

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

sns.set(style="white", color_codes=True)   
from feat_extraction import chi_sq_features,feature_correlations
from plot_methods import PlotGenerator 
if __name__=='__main__':
    train = pd.read_csv("../../resources/data/train/train.csv") # the train dataset is now a Pandas DataFrame
    test = pd.read_csv("../../resources/data/test/test.csv") # the test dataset is now a Pandas DataFrame
    #gen_plots(train,'TARGET','^num_var[0-9]{1,2}$','../../resources/plots/num_vars/','Number of ',True)
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
    #gen_plots(train,'TARGET','^var[0-9]{1,2}$','../../resources/plots/vars/','Value of ',True)
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
    
    PlotGenerator.gen_log_plots(train,'TARGET','../../resources/plots/logs/','Log of ')
    '''
    imp_ent_var16_ult1 i
    imp_op_var39_efect_ult1 i
    imp_op_var39_efect_ult3 i
    imp_op_var40_comer_ult3 d
    imp_op_var40_efect_ult1 i
    imp_op_var40_ult1 d
    imp_op_var41_efect_ult3 i
    imp_sal_var16_ult1 5 or 7
    
    
    saldo_medio_var8_hace2 i 7-9
    saldo_medio_var8_hace3 i 4-6
    saldo_var1 d 7
    saldo_var13_corto i 12-13
    saldo_var13_largo 11-12
    saldo_var14 4
    saldo_var17 d 9-11
    saldo_var25 7.5
    saldo_var26 7.5
    saldo_var30 i 10-14
    saldo_var31 10-11
    saldo_var32 d 6-7.5
    saldo_var40 d 5
    saldo_var42 i 7-12
    saldo_var5 i 6-10
    saldo_var8 i 6-10
    '''
    chi_sq_features.gen_chi_sq_feats(X,y)
    
    #feature_correlations.gen_correlations(train, y)