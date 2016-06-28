import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sklearn



def read_csv(file):
    """
    This function reads the csv file
    :return:
    """
    #file = raw_input("csv file path")
    return pd.read_csv(file)

def Survivors():

    training_data = read_csv('/home/azafar/Projects/predicting_titanic_survivors/data/train.csv')

    print training_data.head()

    # metrics of men who survived

    men_high_fare = training_data[["Fare","Survived", "Sex","Pclass"]][training_data["Sex"]=="male"][training_data["Survived"]==1]

    print "\n metrics of men who survived\n"

    print men_high_fare.describe()

    #metrics of male in pclasses

    men_pclass = training_data["Pclass"][training_data["Sex"]=="male"].describe()

    #total counts for males in different pclass

    men_pclass_counts = training_data["Pclass"][training_data["Sex"] == "male"].value_counts()

    #metrics of male in classes who survived

    men_class_survival = training_data["Pclass"][training_data["Sex"] == "male"][training_data["Survived"]==1].describe()

    # counts of male in pclasses who survived

    men_class_survival_counts = training_data["Pclass"][training_data["Sex"] == "male"][
        training_data["Survived"] == 1].value_counts()

    print "\n metrics of men who survived\n"

    print men_pclass

    print "\n metrics of male in pclasses\n"

    print men_class_survival

    print "\n total counts for males in different pclass\n"

    print men_pclass_counts

    print "\n counts of male in pclasses who survived\n"

    print men_class_survival_counts

    print "\n Men Survival ratio in different pclasses\n"

    men_survival_ratio = (men_class_survival_counts / men_pclass_counts)

    print men_survival_ratio

    # Posterior distribution of men surviving in different classes

    print "\n Posterior distribution of men who survived in different pclasses\n"

    print (men_survival_ratio / men_survival_ratio.sum())

    men_class_survival_counts.plot(kind = 'hist')











run_survivor = Survivors()





