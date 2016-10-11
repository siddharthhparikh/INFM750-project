# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:55:00 2016

@author: Viral-PC
"""

#import matplotlib.pyplot as plt
import csv 
#from GMM_classifier import gmm_classifier
from sklearn.ensemble import RandomForestClassifier
import random
#import numpy as np
import os
os.chdir("C:\Viral\Courses\INFM 750\Data")
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
target_data_list=[]
for key,value in data.iteritems():
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
print rf.score(test_data,test_data_label)
#savetxt('Data/submission2.csv', rf.predict(test_data_list), delimiter=',', fmt='%f')
