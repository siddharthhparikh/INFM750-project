import matplotlib.pyplot as plt
import csv 
import random
import numpy as np
from sklearn.neural_network import MLPClassifier

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
violation_map = {}
i=0
for key,value in data.iteritems():
	random.shuffle(value)
	if len(value) > 15000:
		violation_map[key] = i
		i=i+1
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
	train_data.append([item[0], item[1]])
	train_data_label.append(violation_map[item[2]])
for item in test_data_list:
	test_data.append([item[0], item[1]])
	test_data_label.append(violation_map[item[2]])

X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-2,hidden_layer_sizes=(100, 2000), random_state=1, max_iter=2000, verbose=True)
clf.fit(train_data,train_data_label)
y_predict = clf.predict(test_data)
clf.fit(X,y)
print clf.predict([[2., 2.], [-1., -2.]])
print y_predict
correct = 0
for a,b in zip(y_predict, test_data_label):
	if a == b:
		correct = correct+1

print float(correct)/len(test_data_label)

