import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



def credit()-> pd.DataFrame:
    credit = pd.read_csv("../data/credit_customers.csv")
    return credit

def features(credit: pd.DataFrame)-> pd.DataFrame:
    credit['class'] = credit['class'].map({'good': 0 , 'bad': 1})
    #Converts the target labels ('good', 'bad') into binary numerical values. 
    
    categorical_cols = credit.select_dtypes(include=["object","str"]).columns.to_list()
    #Stores the names of all categorical (object/string) columns in python list.
        
    credit_dum = pd.get_dummies(credit, categorical_cols, drop_first= True)
    #One-hot encoding categorical columns into binary features and drops
    #The first category from each features to avoid redundant dummy vatiables.
    
    return credit_dum

def train_split(credit_dum: pd.DataFrame):
    X = credit_dum.drop(['class'], axis=1)
    y= credit_dum['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    return X_train, X_test, y_train, y_test

def pipeline():
    log_pip = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    return log_pip

