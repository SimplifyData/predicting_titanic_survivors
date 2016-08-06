import dask.dataframe as dd
import matplotlib.pyplot as plt
plt.interactive(False)
import matplotlib
matplotlib.style.use('ggplot')
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

    men_post = (men_survival_ratio / men_survival_ratio.sum())

    print men_post

    # looking at plot of statistical variables

    training_data.Fare.plot.hist()

    # few out lyers in the fare histogram, such as $500 etc since most of the fare lye within $147.

    plt.show()

    training_data["Sex"].value_counts().plot.bar()

    plt.show()

    training_data.Age.plot.hist()

    plt.show()

    training_data['SibSp'].value_counts().plot.bar()

    plt.show()

    training_data['Parch'].value_counts().plot.bar()

    plt.show()


    # group by mean value of the people who survived

    groupby_survived = training_data[["Survived", "Age","Pclass","Fare"]].groupby("Survived").mean()

    print "\n group by value of the people who survived by looking at the price of the ticket"

    print groupby_survived

    # pasangers survival rate accroding to the pclass

    pclass_survived = training_data[["Pclass", "Age", "Fare", "Survived"]].groupby("Pclass").mean()

    print "\n passenger survival rate according to the survival rate"

    print pclass_survived

    sum_survived_bysex = training_data[["Sex","Age","Pclass","Fare","Survived"]].groupby("Sex").mean()

    print "\n total people analyzed by thier sex"

    print sum_survived_bysex

    # total survived by seen by thier sex

    survived_bysex = training_data[["Sex", "Survived"]].groupby("Sex").mean()

    print " \n total survived seen by thier sex"

    print survived_bysex

    #value counts of survivals

    num_ppl_survived = training_data["Survived"].value_counts()

    print "\n value counts of survived people"

    print num_ppl_survived



    #total number of females and males

    sum_f_m = training_data["Sex"].value_counts()

    print "\n total number of males and females"

    print sum_f_m

    # number of females and males surviving

    f_m_survived = (sum_f_m[1] * survived_bysex._get_values[0], sum_f_m[0] * survived_bysex._get_values[1])

    print "\n number of female and males surviving 1) !=survived 2) survived"

    print f_m_survived

    print "\n % of females and males surviving - checked"

    percentage_f_m_survived = f_m_survived[0] / sum_f_m[1], f_m_survived[1] / sum_f_m[0]



























run_survivor = Survivors()





