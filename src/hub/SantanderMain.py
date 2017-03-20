'''
Created on Mar 20, 2017

@author: abhijit.tomar
'''
import pandas as pd
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
from plot_methods import PlotGenerator 
from feat_extraction import chi_sq_features,feature_correlations
if __name__ == '__main__':
    train = pd.read_csv("../../resources/data/train/train.csv") # the train dataset is now a Pandas DataFrame
    test = pd.read_csv("../../resources/data/test/test.csv") # the test dataset is now a Pandas DataFrame
    ''' 
    Generate plots for calculating quantities for variables, where this makes sense. The Y-axis will be the number of customers,
    and the X-axis will be the count/quantity of the variable we are plotting against. The plot will show two bars. Green represents 
    that the value of TARGET is 1 and Blue represents that the value of TARGET is 0  
    '''
    PlotGenerator.gen_plots(train,'TARGET','^num_var[0-9]{1,2}$','../../resources/plots/num_vars/','Number of ')
    # Also generate the kernel density estimation plots for the above condition
    PlotGenerator.gen_plots(train,'TARGET','^num_var[0-9]{1,2}$','../../resources/plots/num_vars/','Number of ',True)
    ''' 
    Generate plots for the values of variables. The Y-axis will be the number of customers, and the X-axis will be the 
    value of the variable we are plotting against. The plot will show two bars. Green represents that the value of TARGET 
    is 1 and Blue represents that the value of TARGET is 0  
    '''
    PlotGenerator.gen_plots(train,'TARGET','^var[0-9]{1,2}$','../../resources/plots/vars/','Value of ')
    # Also generate the kernel density estimation plots for the above condition
    PlotGenerator.gen_plots(train,'TARGET','^var[0-9]{1,2}$','../../resources/plots/vars/','Value of ',True)
    # Add feature that counts the number of zeros in a row
    X = train.iloc[:,:-1]
    y = train.TARGET
    
    X['n0'] = (X==0).sum(axis=1)
    train['n0'] = X['n0']
    '''
    For many variables, all of the values are very close to zero. We will generate Log plots for these for better
    visualization. The Y-axis will be the number of customers, and the X-axis will be the log of the value of the 
    variable we are plotting against
    '''
    PlotGenerator.gen_log_plots(train,'TARGET','../../resources/plots/logs/','Log of ')
    '''
    Generate chi square features. Recall that the chi-square test measures dependence between stochastic
    variables, so using this function "weeds out" the features that are the
    most likely to be independent of class and therefore irrelevant for
    classification.
    '''
    chi_sq_features.gen_chi_sq_feats(X,y,'../../resources/feats/chi2_feats.json')
    '''
    Use chi squared features to compute most important correlations between TARGET and other variables 
    '''
    feature_correlations.gen_correlations(train, '../../resources/feats/chi2_feats.json','../../resources/feats/unique_important_corrs.csv')