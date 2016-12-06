# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 15:11:21 2016

@author: Viral-PC
"""

import csv
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import tree

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print(cm)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, round(cm[i, j]*100,2),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    else:
        print('Confusion matrix, without normalization')
        print(cm)    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def r2score(y_actual, y_pred):
    tss = 0
    mse = 0
    for a,b in zip(y_actual['volumetype'], y_pred):
        #print a, b[0]
        mse += (a-b[0])*(a-b[0])

    mse = mse/len(y_pred)
    avg = 0
    for a in y_actual['volumetype']:
        avg += a
    avg = avg/len(y_actual['volumetype'])
    for a in y_actual['volumetype']:
        tss += (a-avg)*(a-avg)  
    r2 = 1-len(y_pred)*(mse/tss)
    return r2

def score(y_pred, y_actual):
    score = 0
    for a,b in zip(y_pred, y_actual['volumetype']):
        if a == b:
            score += 1
    return float(score)/len(y_pred)

data = {}
violation = {}
i=0
#with open('datasets/data_boston-2.csv', 'r') as csvfile:
with open('C:\Viral\Courses\INFM 750\Data\data_boston.csv', 'r') as csvfile:
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
                data[row[12]] = [int(row[12]), float(row[17]), float(row[18]), float(row[21]), float(row[20]),1]

##normalizing the volume of violations with the population of zipcodes
dat = {}

##creating classes of volume of violations
for key, value in data.iteritems():
    if value[-1] > 2000:
        dat[key] = data[key]
        dat[key][-1] = value[-1]/value[-2]
        if value[-1] > 1.53:
            dat[key].append(2)
        elif value[-1] < 1.53 and value[-1] > 1.18:
            dat[key].append(2)
        elif value[-1] < 1.18 and value[-1] > 0.83:
            dat[key].append(2)
        elif value[-1] < 0.83 and value[-1] > 0.49:
            dat[key].append(1)
        else:
            dat[key].append(0)
        
j=0
err = 0
err_svm = 0
err_dt = 0
err_rf = 0
data_list = []
for key, value in dat.iteritems():
        data_list.append(value)

y_actual_conf = []
y_pred_conf = []    
cnf_matrix = [[0,0,0],[0,0,0],[0,0,0]]    
while j<10000:
    random.shuffle(data_list);
    
    df_data_list = pd.DataFrame(data_list)
    df_data_list.columns = ['zipcode','medianincome','collegedegree','houseowner','population','volumeofviolations','volumetype']

#    df_data_list.to_csv("boston_socioeco_info.csv")
    for col in df_data_list:
        if col != 'zipcode' and col != 'volumetype':
            df_data_list[col] = (df_data_list[col] - df_data_list[col].mean())/df_data_list[col].std(ddof=0)
  #  print df_data_list
    
##create train and test df 70:30 

    train, test = train_test_split(df_data_list, test_size = 0.3)
    train_data_label = train[['volumetype']]
    train_data_list = train[['medianincome','collegedegree','houseowner']]
    test_data_label = test[['volumetype']]
    test_data_list = test[['medianincome','collegedegree','houseowner']]

##logistic regression
    logistic = linear_model.LogisticRegression()
    logistic.fit(train_data_list, train_data_label['volumetype'])
    predict = logistic.predict(test_data_list)
    err += score(predict,test_data_label)
    
##SVM
    svm_model = svm.SVC()
    svm_model.fit(train_data_list, train_data_label['volumetype'])
    y_svm_predict = svm_model.predict(test_data_list)
    cnf_matrix = cnf_matrix + confusion_matrix(test_data_label, y_svm_predict, labels=[0,1,2])
    err_svm += score(y_svm_predict,test_data_label)
    
#Decision Tree
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(train_data_list, train_data_label['volumetype'])
    y_dt_predict = dt.predict(test_data_list)
    err_dt += score(y_dt_predict,test_data_label)

#Random Forest    
    rf = RandomForestClassifier()
    rf.fit(train_data_list, train_data_label['volumetype'])
    y_rf_predict = rf.predict(test_data_list)
    err_rf += score(y_rf_predict,test_data_label)
        
    j=j+1

#Confusion Matrix of SVM Model (Best model)
class_names=["low","medium","high"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

##Accuracy of the models run
print "Logistic"
print err/j
print "SVM"
print err_svm/j
print "Decision Tree"
print err_dt/j
print "Random forest"
print err_rf/j
