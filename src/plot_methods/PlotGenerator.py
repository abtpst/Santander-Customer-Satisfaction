'''
Created on Mar 20, 2017

@author: abhijit.tomar
'''
import re
import numpy as np
import seaborn as sns
sns.set(style="white", color_codes=True)   
import matplotlib.pyplot as plt

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
                print('Too many zeros in '+col)

def gen_log_plots(in_df,target_col,save_dir,x_label):                 
    
    cols = in_df.columns
    
    for col in cols:
        in_df[col]=in_df[col].dropna()
        try:
            in_df[col].replace(0, np.nan).dropna().map(np.log).hist(by=in_df['TARGET'],bins=1000,log=True)
                #histtype : {'bar', 'barstacked', 'step',  'stepfilled'}, optional
            plt.xlabel(x_label+col)
            plt.ylabel('Number of customers in train')
            plt.savefig(save_dir+col+'.png')
            plt.close()
            print(save_dir+col+'.png')
        except :
            pass