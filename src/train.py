import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib



def load_data()-> pd.DataFrame:
    credit = pd.read_csv("../data/credit_customers.csv")
    return credit

def features(credit: pd.DataFrame)-> pd.DataFrame:
    credit['class'] = credit['class'].map({'good': 0 , 'bad': 1})
    #Converts the target labels ('good', 'bad') into binary numerical values. 
    
    categorical_cols = credit.select_dtypes(include=["object","str"]).columns.to_list()
    #Stores the names of all categorical (object/string) columns in python list.
        
    credit_dum = pd.get_dummies(credit, columns=categorical_cols, drop_first= True)
    #One-hot encoding categorical columns into binary features and drops
    #The first category from each features to avoid redundant dummy vatiables.
    
    return credit_dum

def train_split(credit_dum: pd.DataFrame):
    X = credit_dum.drop(['class'], axis=1) 
    #Drops the target variable 'class' to create the feature matrix (X).
    
    y = credit_dum['class']
    # Extracts the target column ('class') for model training
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.20
        )
    #Splits the data into training (75%) and testing (25%) sets.
    #Random_state=42 ensures the split is reproducible across runs.
    return X_train, X_test, y_train, y_test

def build_pipeline() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000,random_state=42))
    ])

def train_model(pipeline: Pipeline, X_train, y_train)-> Pipeline:
    Pipeline.fit(X_train, y_train)
    return pipeline
    
