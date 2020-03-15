# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:34:28 2020

@author: Gavin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn import preprocessing


ave_monthspend = pd.read_csv('AW_AveMonthSpend.csv')
bike_buyer = pd.read_csv('AW_BikeBuyer.csv')
work_cust = pd.read_csv('WorksCusts.csv')

work_cust.columns = [str.replace('-', '_') for str in work_cust.columns]
work_cust.drop_duplicates(subset = 'CustomerID', keep = 'first', inplace = True)

work_cust['BikeBuyer'] = bike_buyer['BikeBuyer']
work_cust['AveMonthSpend'] = ave_monthspend['AveMonthSpend']
work_cust.head().transpose()

print(datasetWorkCusts.shape)
print(datasetWorkCusts.CustomerID.unique().shape)
work_cust.Gender.value_counts()

work_cust.drop(['Title','MiddleName','Suffix', 'AddressLine2'], axis = 1, inplace = True)

def calculate_age(end):
    r = relativedelta(pd.to_datetime('now'), pd.to_datetime(end)) 
    return '{}'.format(r.years)


work_cust['Age'] = work_cust['BirthDate'].apply(calculate_age)

# Convert age to integer
work_cust['Age'] = work_cust['Age'].astype('int64')

def plot_box(work_cust, cols, col_x = 'BikeBuyer'):
    for col in cols:
        sns.set_style('whitegrid')
        sns.boxplot(col_x, col, data=work_cust)
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()

num_cols = ['YearlyIncome','NumberCarsOwned']
plot_box(work_cust, num_cols)


cat_cols = ['Occupation','Gender','MaritalStatus']

work_cust['dummy'] = np.ones(shape = work_cust.shape[0])
for col in cat_cols:
    print(col)
    counts = work_cust[['dummy', 'BikeBuyer', col]].groupby(['BikeBuyer', col], as_index = False).count()
    temp = counts[counts['BikeBuyer'] == 0][[col, 'dummy']]
    _ = plt.figure(figsize = (10,4))
    plt.subplot(1, 2, 1)
    temp = counts[counts['BikeBuyer'] == 0][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n no bike purchase')
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    temp = counts[counts['BikeBuyer'] == 1][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n customer bought bike')
    plt.ylabel('count')
    plt.show()
    

work_cust.to_csv('work_cust.csv')