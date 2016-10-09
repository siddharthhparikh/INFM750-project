
import matplotlib.pyplot as plt
import csv 
import numpy as np
from scipy import linalg
import itertools
from sklearn import mixture
import math
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import multivariate_normal
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

def gaussian_value(x,mean,covariance):
	var = multivariate_normal(mean, covariance)
	return var.pdf(x)

class gmm_classifier(BaseEstimator, ClassifierMixin):
	def __init__(self, a=None):
		self.a = a
	
	def fit(self, X, y):
		#print X, violations
		X, y = check_X_y(X, y)
		self.classes_ = unique_labels(y)
		data = self.make_dictionary(X, y)
		self.m = self.generate_model(data)
		return self


	def make_dictionary(self,X,violations):
		data = {}
		for cord,viol in zip(X,violations):
			if data.has_key(viol):
				data[viol].append([float(x) for x in cord])
				#data[viol].append([float(cord[0]), float(cord[1])])
			else:
				data[viol] = [[float(x) for x in cord]]
				#data[viol] = [[float(cord[0]), float(cord[1])]]
		return data

	def generate_model(self, data):
		m = {}
		for key,value in data.iteritems():
			random.shuffle(value)
			m[key] = self.fit_samles(np.array(value))
		return m

	def fit_samles(self, samples):
		gmix = mixture.BayesianGaussianMixture(n_components=len(samples), covariance_type='full', max_iter=1000, verbose=1).fit(samples)
		return gmix
	
	def predict(self, X, y=None):
		violations = list()
		# Check is fit had been called
		check_is_fitted(self, ['m'])
		# Input validation
		X = check_array(X)
		for coordinate in X:
			max_val = 0
			max_violation = ''
			for k, val in self.m.iteritems():
				a = val.predict(coordinate)
				temp = gaussian_value(coordinate, val.means_[a][0], val.covariances_[a][0]) 
				if (temp > max_val):
						max_violation = k
						max_val = temp
			violations.append(max_violation)
		return np.array(violations)
		

