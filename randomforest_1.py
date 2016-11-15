# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:55:00 2016
@author: Viral-PC
"""

#import matplotlib.pyplot as plt
import csv 
#from GMM_classifier import gmm_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
#from sklearn.neural_network import MLPClassifier
import random
#import numpy as np
import os
data = {}
with open('data_boston.csv', 'r') as csvfile:
	csvfile.readline()
	file = csv.reader(csvfile, delimiter=',')
	for row in file:
		if data.has_key(row[5]):
			data[row[5]].append([float(row[14]), float(row[15]), row[5]])
		else:
			data[row[5]] = [[float(row[14]), float(row[15]), row[5]]]

test_data_list = []
train_data_list = []

for key,value in data.iteritems():
     value=random.sample(value,len(value))
     if len(value) > 15000:
		for val in value[:15000]:
			train_data_list.append(val)   
		for val in value[15000:19000]:
			test_data_list.append(val)
   
train_data = list()
train_data_label = list()
test_data = list()
test_data_label = list()

random.shuffle(train_data_list)
random.shuffle(test_data_list)

for item in train_data_list:
	train_data.append([item[0], item[1]])
	train_data_label.append(item[2])
for item in test_data_list:
	test_data.append([item[0], item[1]])
	test_data_label.append(item[2])

#print len(train_data), len(train_data_label), len(train_data_list)

#rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)
rf = RandomForestClassifier()
rf.fit(train_data, train_data_label)
#print rf.predict(test_data)
print "Random forest"
print rf.score(test_data,test_data_label)
#savetxt('Data/submission2.csv', rf.predict(test_data_list), delimiter=',', fmt='%f')

#Logistic Regression
logistic = linear_model.LogisticRegression()
logistic.fit(train_data, train_data_label)

print "Logistic Regression"
print logistic.score(test_data,test_data_label)

#SVM

from sklearn import svm
#svm_model = svm.SVC(decision_function_shape='ovo')
svm_model = svm.SVC()
svm_model.fit(train_data, train_data_label)

print "SVM"
print svm_model.score(test_data,test_data_label)

#Decision Tree
from sklearn import tree
#import pydotplus
dt = tree.DecisionTreeClassifier()
dt = dt.fit(train_data, train_data_label)

print "Decision Tree"
print dt.score(test_data,test_data_label)
"""
from IPython.display import Image  
dot_data = tree.export_graphviz(dt, out_file=None, 
                         feature_names=train_data,  
                         class_names=train_data_label,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  
"""
#Neural Network
"""
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn.fit(train_data, train_data_label)

print nn.score(test_data, test_data_label)
"""
