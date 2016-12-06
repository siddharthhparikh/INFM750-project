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
from sklearn.neural_network import MLPClassifier
import random
#import numpy as np
import os
data = {}

import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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

with open('datasets/data_boston-2.csv', 'r') as csvfile:
	csvfile.readline()
	file = csv.reader(csvfile, delimiter=',')
	for row in file:
		if row[17] != '' and row[18] != '':
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
#print rf.score(test_data,test_data_label)

y_rf_predict = rf.predict(test_data)
print "After predict"
#print y_rf_predict
correct = 0
for a,b in zip(y_rf_predict, test_data_label):
	if a == b:
		correct = correct+1

print float(correct)/len(test_data_label)

#savetxt('Data/submission2.csv', rf.predict(test_data_list), delimiter=',', fmt='%f')
cnf_matrix = confusion_matrix(test_data_label, y_rf_predict,labels=["Improper storage trash: res", "Overgrown Weeds On Property", "Overfilling of barrel/dumpster", "Failure clear sidewalk - snow"])
# Plot non-normalized confusion matrix
plt.figure()
class_names = ["Improper storage trash: res", "Overgrown Weeds On Property", "Overfilling of barrel/dumpster", "Failure clear sidewalk - snow"]
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()
exit()
#Logistic Regression
logistic = linear_model.LogisticRegression()
logistic.fit(train_data, train_data_label)

print "Logistic Regression"
#print logistic.score(test_data,test_data_label)
y_logistic_predict = logistic.predict(test_data)
print "After predict"
correct = 0
for a,b in zip(y_logistic_predict, test_data_label):
	if a == b:
		correct = correct+1

exit()
print float(correct)/len(test_data_label)
#SVM

from sklearn import svm
#svm_model = svm.SVC(decision_function_shape='ovo')
svm_model = svm.SVC()
svm_model.fit(train_data, train_data_label)

print "SVM"
#print svm_model.score(test_data,test_data_label)
y_svm_predict = svm_model.predict(test_data)
print "After predict"
#print y_svm_predict
correct = 0
for a,b in zip(y_svm_predict, test_data_label):
	if a == b:
		correct = correct+1

print float(correct)/len(test_data_label)

#Decision Tree
from sklearn import tree
#import pydotplus
dt = tree.DecisionTreeClassifier()
dt = dt.fit(train_data, train_data_label)

print "Decision Tree"
#print dt.score(test_data,test_data_label)

y_dt_predict = dt.predict(test_data)
print "After predict"
#print y_dt_predict
correct = 0
for a,b in zip(y_dt_predict, test_data_label):
	if a == b:
		correct = correct+1

print float(correct)/len(test_data_label)

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