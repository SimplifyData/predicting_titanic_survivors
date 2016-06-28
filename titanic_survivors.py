import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sklearn



def read_csv(file):
    """
    This function reads the csv file
    :return:
    """
    return pd.read_csv(file)

def Survivors():

    training_data = read_csv('/home/azafar/Projects/predicting_titanic_survivors/data/train.csv')

    print training_data



run_survivor = Survivors()





