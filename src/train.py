import pandas as pd
import numpy as np


def credit()-> pd.DataFrame:
    credit = pd.read_csv("../data/credit_customers.csv")
    return credit
