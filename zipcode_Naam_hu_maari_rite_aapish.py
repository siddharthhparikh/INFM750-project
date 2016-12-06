def calculate_income_range(income):
    """
    if income < 25000:
        return 0
    elif income < 50000:
        return 1
    elif income < 75000:
        return 2
    else:
        return 3
    """
    return int(income/10000)

import csv
import random
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
import pandas as pd
#import numpy as np
data = {}
violation = {}
i=0
with open('datasets/data_boston-2.csv', 'r') as csvfile:
#with open('C:\Viral\Courses\INFM 750\Data\data_boston.csv', 'r') as csvfile:
    csvfile.readline()
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        if row[17] != '' and row[18] != '':
            if not violation.has_key(row[5]):
                violation[row[5]] = i
                i=i+1
            if data.has_key(row[12]):
                data[row[12]][-1] = data[row[12]][-1] + 1
            else:
                data[row[12]] = [int(row[12]), float(row[17]), float(row[18]), float(row[21]), float(row[20]), 1]

##normalizing the volume of violations with the population of zipcodes
dat = {}
for key, value in data.iteritems():
    if value[-1] > 2000:
        dat[key] = data[key]
        dat[key][-1] = value[-1]/value[-2]


j=0
score = 0
err = 0

data_list = []
for key, value in dat.iteritems():
        data_list.append(value)

from sklearn.linear_model import Ridge
#print data_list
<<<<<<< HEAD:zipcode.py
=======

import math
def r2score(y_actual, y_pred):
    tss = 0
    mse = 0
    for a,b in zip(y_actual[0], y_pred):
        mse += (a-b[0])*(a-b[0])

    mse = mse/len(y_pred)
    #print "len(y_actual[0]) = ", len(y_actual[0])
    avg = 0
    for a in y_actual[0]:
        avg += a
    avg = avg/len(y_actual[0])
    #print "avg = ", avg
    for a in y_actual[0]:
        #print a
        tss += (a-avg)*(a-avg)

    r2 = 1-(len(y_actual[0])*mse/tss)
    return r2
    
>>>>>>> 6c1fb023788e784aa005e47e44649d928d4ef8a4:zipcode_Naam_hu_maari_rite_aapish.py
while j<1000:
    test_data_list = []
    test_data_label = []
    train_data_list = []
    train_data_label = []
    i=0

    random.shuffle(data_list);
    
    df_data_list = pd.DataFrame(data_list)
    df_data_list.columns = ['zipcode','medianincome','collegedegree','houseowner','population','volumeofviolations']
    ##removing population column from the df
    df_data_list = df_data_list.drop('population', 1)
    for col in df_data_list:
        if col != 'zipcode':
            df_data_list[col] = (df_data_list[col] - df_data_list[col].mean())/df_data_list[col].std(ddof=0)
    
##create test and train df 

    train, test = train_test_split(df_data_list, test_size = 0.2)
#    print test
    train_data_label = train[['volumeofviolations']]
    train_data_list = train[['medianincome','collegedegree','houseowner']]
    test_data_label = test[['volumeofviolations']]
    test_data_list = test[['medianincome','collegedegree','houseowner']]

 #   print train
    
    regr = linear_model.LinearRegression()
    regr.fit(train_data_list, train_data_label)
    err = err + r2_score(test_data_label, regr.predict(test_data_list)); 
    # Explained variance score: 1 is perfect prediction
    score = score + regr.score(test_data_list, test_data_label)

#    for value in data_list:
#        if i>20:
#            test_data_list.append(value[:-2])
#            test_data_label.append(value[-1])
#        else:
#            train_data_list.append(value[:-2])
#            train_data_label.append(value[-1])
#        i=i+1
#    df_test_data_list = pd.DataFrame(test_data_list)
##    print df_test_data_list
#    df_test_data_label = pd.DataFrame(test_data_label)
#    df_train_data_list = pd.DataFrame(train_data_list)
#    #print df_train_data
#    df_train_data_label = pd.DataFrame(train_data_label)
#    for col in df_test_data_list:
#        df_test_data_list[col] = (df_test_data_list[col] - df_test_data_list[col].mean())/df_test_data_list[col].std(ddof=0)
#    for col in df_test_data_label:
#        df_test_data_label[col] = (df_test_data_label[col] - df_test_data_label[col].mean())/df_test_data_label[col].std(ddof=0)
#    for col in df_train_data_list:
#        df_train_data_list[col] = (df_train_data_list[col] - df_train_data_list[col].mean())/df_train_data_list[col].std(ddof=0)
#    for col in df_train_data_label:
#        df_train_data_label[col] = (df_train_data_label[col] - df_train_data_label[col].mean())/df_train_data_label[col].std(ddof=0)

##polynomial regression
#    poly = PolynomialFeatures(degree=2)
#    df_train_data_list_poly = poly.fit_transform(df_train_data_list)
#    df_test_data_list_poly = poly.fit_transform(df_test_data_list)
#    
    for value in data_list:
        if i>18:
            test_data_list.append(value[:-2])
            test_data_label.append(value[-1])
        else:
            train_data_list.append(value[:-2])
            train_data_label.append(value[-1])
        i=i+1
    df_test_data_list = pd.DataFrame(test_data_list)
#    print df_test_data_list
    df_test_data_label = pd.DataFrame(test_data_label)
    df_train_data_list = pd.DataFrame(train_data_list)
    #print df_train_data
    df_train_data_label = pd.DataFrame(train_data_label)
    for col in df_test_data_list:
        df_test_data_list[col] = (df_test_data_list[col] - df_test_data_list[col].mean())/df_test_data_list[col].std(ddof=0)
    for col in df_test_data_label:
        df_test_data_label[col] = (df_test_data_label[col] - df_test_data_label[col].mean())/df_test_data_label[col].std(ddof=0)
    for col in df_train_data_list:
        df_train_data_list[col] = (df_train_data_list[col] - df_train_data_list[col].mean())/df_train_data_list[col].std(ddof=0)
    for col in df_train_data_label:
        df_train_data_label[col] = (df_train_data_label[col] - df_train_data_label[col].mean())/df_train_data_label[col].std(ddof=0)

#    print df_test_data_list
#    print df_test_data_label
#    print df_train_data_list
#    print df_train_data_label
#    test_data = pd.Dataframe.from_dict(test_data_list,orient='index')
#    test_label = pd.Dataframe.from_dict(test_data_label,orient='index')
#print "test :"; print test_data_list; print "train : "; print train_data_list;
    
#    regr = linear_model.LinearRegression()
#    regr.fit(df_train_data_list, df_train_data_label)
#    err = err + r2_score(df_test_data_label, regr.predict(df_test_data_list)); #print r2_score(test_data_label, regr.predict(test_data_list))
#    # Explained variance score: 1 is perfect prediction
#    score = score + regr.score(df_test_data_list, df_test_data_label)

    regr = linear_model.LinearRegression()
    regr.fit(df_train_data_list, df_train_data_label)
    y_pred = regr.predict(df_test_data_list)
    y_actual = df_test_data_label
    #print len(y_pred),len(y_actual)
    err = err + r2score(df_test_data_label, regr.predict(df_test_data_list)); #print r2_score(test_data_label, regr.predict(test_data_list))
    # Explained variance score: 1 is perfect prediction
    score = score + regr.score(df_test_data_list, df_test_data_label)

    j=j+1
print err/j