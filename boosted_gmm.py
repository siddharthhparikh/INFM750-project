import matplotlib.pyplot as plt
import csv 
from GMM_classifier import gmm_classifier
from sklearn.ensemble import AdaBoostRegressor
import random
import numpy as np

data = {}
with open('datasets/data_boston.csv', 'r') as csvfile:
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
	if len(value) > 15000:
		for val in value[:15000]:
			train_data_list.append(val)
		for val in value[15000:19000]:
			test_data_list.append(val)

del data
"""
train_data = np.array([[train_data_list[0][0], train_data_list[0][1]]])
train_data_label = np.array([[train_data_list[0][2]]])
test_data = np.empty(4000)
test_data_label = np.empty(4000)
"""
"""
train_data = np.zeros((15000,2))
train_data_label = []
test_data = np.zeros((4000,2))
test_data_label = []
i=0
for item in train_data_list[0]:
	train_data[i,0] = item[0] 
	train_data[i,1] = item[1]
	train_data_label.append(item[2])
	i=i+1

i=0
for item in test_data_list[0]:
	test_data[i,0] = item[0] 
	test_data[i,1] = item[1]
	test_data_label.append(item[2])
	i=i+1

del test_data_list
del train_data_list
"""

d = {
	'Improper storage trash: res' : 0,
	'Overgrown Weeds On Property' : 1,
	'Failure clear sidewalk - snow' : 2,
	'Overfilling of barrel/dumpster' : 3
}

train_data = list()
train_data_label = list()
test_data = list()
test_data_label = list()

random.shuffle(train_data_list)
random.shuffle(test_data_list)

for item in train_data_list:
	train_data.append([item[0], item[1]])
	train_data_label.append(d[item[2]])
for item in test_data_list:
	test_data.append([item[0], item[1]])
	test_data_label.append(d[item[2]])

print len(train_data), len(train_data_label), len(train_data_list)
regr = gmm_classifier()
regr.fit(train_data,train_data_label)
y_predict = regr.predict(test_data)
print y_predict
correct = 0
for a,b in zip(y_predict, test_data_label):
	if a == b:
		correct = correct+1

print float(correct)/len(test_data_label)

