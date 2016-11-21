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
import numpy as np 
data = {}
violation = {}
i=0
with open('datasets/data_boston-2.csv', 'r') as csvfile:
	csvfile.readline()
	file = csv.reader(csvfile, delimiter=',')
	for row in file:
		if row[17] != '' and row[18] != '':
			if not violation.has_key(row[5]):
				violation[row[5]] = i
				i=i+1
			if data.has_key(row[5]):
				data[row[5]].append([float(row[14]), float(row[15]), float(row[18]), float(row[21]), violation[row[5]], calculate_income_range(float(row[17]))])
			else:
				data[row[5]] = [[float(row[14]), float(row[15]), float(row[18]), float(row[21]), violation[row[5]], calculate_income_range(float(row[17]))]]

test_data_list = []
train_data_list = []

for key,value in data.iteritems():
	random.shuffle(value)
	if len(value) > 19000:
		for val in value[:15000]:
			train_data_list.append(val)   
		for val in value[15000:19000]:
			test_data_list.append(val)

del data

train_data = list()
train_data_label = list()
test_data = list()
test_data_label = list()

random.shuffle(train_data_list)
random.shuffle(test_data_list)

for item in train_data_list:
	train_data.append([item[0], item[1], item[2], item[3], item[5]])
	train_data_label.append(item[4])
for item in test_data_list:
	test_data.append([item[0], item[1], item[2], item[3], item[5]])
	test_data_label.append(item[4])

"""
regr = linear_model.LinearRegression()
regr.fit(train_data, train_data_label)

from sklearn.metrics import r2_score

y_predict = regr.predict(test_data)
print r2_score(test_data_label, y_predict)
"""

logistic = linear_model.LogisticRegression()
logistic.fit(train_data, train_data_label)

print logistic.coef_
print "Logistic Regression"
print logistic.score(test_data,test_data_label)
