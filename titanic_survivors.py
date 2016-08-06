import dask.dataframe as dd
import matplotlib.pyplot as plt
plt.interactive(False)
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import re
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

    print "\n number of female and males surviving"

    print f_m_survived

    print "\n % of females and males surviving - checked"

    percentage_f_m_survived = f_m_survived[0] / sum_f_m[1], f_m_survived[1] / sum_f_m[0]

    print percentage_f_m_survived

    # filling the null values

    print "\n any with more than 0% null values? - Yes : 1) Age, 2) Cabin 3) Embarked"

    print training_data.ix[:, :].isnull().mean()

    print "\n filling these nulls values with medians"

    median_age = training_data["Age"].median()

    print "\n median age"

    print median_age

    #filling in null values for Age

    training_data["Age"].fillna(median_age, inplace= True)

    print "\n tail of Age column"

    print training_data["Age"].tail()

    print "\n imputing missing values in Cabin with No Cabin"

    training_data["Cabin"].fillna(0, inplace= True)

    print "\n filling Embarked nulls with top value"

    print "\n Embarked top value"\

    print "\n Mode of Embarked"

    print training_data["Embarked"].mode()

    print "\n Embarked description"

    print training_data["Embarked"].describe()

    training_data["Embarked"].fillna(training_data["Embarked"].mode(), inplace= True)

    print "\n Females survival analysis"

    f_survival_view = training_data[["Pclass", "Sex", "Age", "SibSp", "Fare", "Survived", "Embarked"]][
        training_data["Sex"] == "female"].groupby("Survived").describe()

    print f_survival_view

    print "\n Females with higher chance of survival: \n 1) (> 90%) 1 or less Siblings \n 2) Fare atleast (> 75%) $13 or higher \n 3) Passanger (75%) class 2 or lower \n 4) Age around 19-37 with 0 sibblings"

    print "\n checking the tail of the df"

    print training_data.tail()

    # converting categories in binary

    print "\n Converting categories into binary values \n male = 0 , female = 1"


    training_data.loc[training_data["Sex"]=="male", "Sex"] = 0

    training_data.loc[training_data["Sex"] == "female", "Sex"] = 1

    #print training_data["Sex"]

    print training_data["Sex"].unique()

    print "\n Embarked : categories \n 1) S = 0 , \n 2) C= 1, \n 3) Q = 2, 4) nan = 3"

    print training_data["Embarked"].unique()


    training_data.loc[training_data["Embarked"] == "S", "Embarked"] = 0

    training_data.loc[training_data["Embarked"] == "C", "Embarked"] = 1

    training_data.loc[training_data["Embarked"] == "Q", "Embarked"] = 2

    print training_data["Embarked"].unique()

    print training_data["Embarked"].value_counts()

    #cabin binary transformation O^n
    """

    for num in range(0,len(training_data["Cabin"].unique())):
        print "size" + str(training_data["Cabin"].unique().shape[0])

        print num

        line = str(training_data["Cabin"].unique()[num])

        print line

        cabin = re.findall("[A-Z]+",line)

        print cabin

        if len(cabin) >= 1:
            training_data.loc[training_data["Cabin"] == line,"Cabin"] = cabin[0]
            #training_data["Cabin"].replace(line,cabin[0])
    """

    print training_data["Cabin"].unique()

    #recursive - cleaning Cabin data to just the first Alphabets
    count = 0

    while count != len(training_data["Cabin"].unique()):
        line = training_data["Cabin"].unique().tolist()[count]

        #print line

        if count == len(training_data["Cabin"].unique()):
            #print str(count) + " break"
            break

        elif line == 0 or len(line) == 1:
            count +=1

            #print "add count" + str(count)

        else:
            cabin = re.findall("[A-Z]+", line)

            #print cabin

            training_data.loc[training_data["Cabin"] == str(line), "Cabin"] = cabin[0]

            count = 0
    print "Cabin binary data: \n  1) 0 = 0 ,\n 2) A = 1 ,\n 3) B = 2 , \n 4) C = 3 ,\n 5) D = 4 ,\n 6) E = 5 ,\n 7) F = 6 ,\n 8) G = 7 ,\n 9) T = 8 "

    training_data.loc[training_data["Cabin"] == "A", "Cabin"] = 1

    training_data.loc[training_data["Cabin"] == "B", "Cabin"] = 2

    training_data.loc[training_data["Cabin"] == "C", "Cabin"] = 3

    training_data.loc[training_data["Cabin"] == "D", "Cabin"] = 4

    training_data.loc[training_data["Cabin"] == "E", "Cabin"] = 5

    training_data.loc[training_data["Cabin"] == "F", "Cabin"] = 6

    training_data.loc[training_data["Cabin"] == "G", "Cabin"] = 7

    training_data.loc[training_data["Cabin"] == "T", "Cabin"] = 8

    print training_data.head()

    print "\n re Females survival analysis mean "

    f_survival_view = training_data[["Pclass", "Sex", "Age", "SibSp", "Fare", "Survived", "Embarked", "Cabin"]][
        training_data["Sex"] == 1].groupby("Survived").mean()

    print f_survival_view


run_survivor = Survivors()





