import pandas as pd
import numpy as np


def credit()-> pd.DataFrame:
    credit = pd.read_csv("../data/credit_customers.csv")
    return credit

def features(credit: pd.DataFrame):
    credit['Class'] = credit['Class'].map({'good': 0 , 'bad': 1})
    #Converts the target labels ('good', 'bad') into binary numerical values. 
    
    categorical_cols = credit.select_dtypes(include=["object","str"]).columns.to_list()
    #Stores the names of all categorical (object/string) columns in python list.
        
    credit_dum = pd.get_dummies(credit, categorical_cols, drop_first= True)
    #One-hot encoding categorical columns into binary features and drops
    #The first category from each features to avoid redundant dummy vatiables.
    
    return credit_dum