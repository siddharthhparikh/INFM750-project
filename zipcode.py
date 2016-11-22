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
from sklearn.metrics import r2_score

import numpy as np 
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
				data[row[12]] = [float(row[18]), float(row[21]), float(row[17]), float(row[20]), 1]

##normalizing the volume of violations with the population of zipcodes
for key, value in data.iteritems():
    value[-1] = value[-1]/value[-2]


j=0
score = 0
err = 0

data_list = []
for key, value in data.iteritems():
		data_list.append(value)

from sklearn.linear_model import Ridge
#print data_list
while j<10000:
	test_data_list = []
	test_data_label = []
	train_data_list = []
	train_data_label = []
	i=0
	
	random.shuffle(data_list)
	for value in data_list:
		if i>22:
			test_data_list.append(value[:-1])
			test_data_label.append(value[-1])
		else:
			train_data_list.append(value[:-1])
			train_data_label.append(value[-1])
		i=i+1

	regr = linear_model.LinearRegression()
	regr.fit(train_data_list, train_data_label)

	err = err + r2_score(test_data_label, regr.predict(test_data_list))
	# Explained variance score: 1 is perfect prediction
	score = score + regr.score(test_data_list, test_data_label)
	
	"""
	clf = Ridge()
	clf.fit(train_data_list, train_data_label)
	err = err + r2_score(test_data_label, clf.predict(test_data_list))
	"""
	j=j+1

print err/j