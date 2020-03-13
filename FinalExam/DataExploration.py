# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:32:07 2020

@author: Gavin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('AW_AveMonthSpend.csv')

print(dataset.columns)

X = dataset.iloc[:, 1].values
rows = X.shape[0]

def calculateMinMax(): 
    minimum = 2000000
    maximum = 0
    for value in np.nditer(X):
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    print(minimum)
    print(maximum)
    
def calculateMean():
    total = 0
    for value in np.nditer(X):
        total += value
    print(total/rows)
    
def calculateMedian():
    print(np.median(X))
    
def calculateSTD():
    print(np.std(X))
    
calculateMinMax()
calculateMean()
calculateMedian()
calculateSTD()


dataset2 = pd.read_csv('AW_BikeBuyer.csv')
Y = dataset2.iloc[:, 1].values
print(type(Y))

def calculateDistribution():
    notBought = 0
    Bought = 0
    for value in np.nditer(Y):
        if (value == 1):
            Bought += 1
        else:
            notBought += 1
    print(notBought)
    print(Bought)
    
calculateDistribution()

dataset3 = pd.read_csv('ADvWorksCusts.csv')
print(type(dataset3))
X = dataset3.iloc[:,15]
Y = dataset3.iloc[:,22].values

def problem_7():
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelEncoder_X = LabelEncoder()
    Xcat = labelEncoder_X.fit_transform(X)
    
    Zero = []
    One = []
    Two = []
    Three = []
    Four = []
    
    count = 0
    for value in np.nditer(Xcat):
        if (value == 0):
            Zero.append(Y[count])
        elif (value == 1):
            One.append(Y[count])
        elif (value == 2):
            Two.append(Y[count])
        elif (value == 3):
            Three.append(Y[count])
        else:
            Four.append(Y[count])
        count += 1
            
    print(np.median(Zero))
    print(np.median(One))
    print(np.median(Two))
    print(np.median(Three))
    print(np.median(Four))
    
problem_7()

