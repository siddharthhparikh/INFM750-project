import matplotlib.pyplot as plt
import csv 

dict = {}
with open('datasets/data_boston.csv', 'r') as csvfile:
	csvfile.readline()
	file = csv.reader(csvfile, delimiter=',')
	for row in file:
		if dict.has_key(row[5]):
			dict[row[5]].append([float(row[14]), float(row[15])])
		else:
			dict[row[5]] = [[float(row[14]), float(row[15])]]
		
import numpy as np
from scipy import linalg
import itertools
import matplotlib as mpl

#from sklearn import mixture
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'green', 'red', 'black', 'orange', 'blue'])

def plot_results(X, Y_, means, covariances, index, title):
    print type(X)
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i,0], X[Y_ == i,1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.subplot(2, 1, 1)
    plt.scatter(X[:,0], X[:,1])

from sklearn import mixture
def fit_samples(samples):
	gmix = mixture.BayesianGaussianMixture(n_components=100, covariance_type='full', max_iter=1000, verbose=1)
	gmix.fit(samples)
	return gmix

models = {}
test_data = {}
import random
for key,value in dict.iteritems():
	if len(value) > 15000:
		random.shuffle(value)
		models[key] = fit_samples(np.array(value[:15000]))
		test_data[key] = np.array(value[15000:19000])

import math
def gaussian_value(x,mean,covariance):
	return (math.exp(-(math.pow((x[0]-mean[0]),2)/covariance[0]+math.pow((x[1]-mean[1]),2)/covariance[1])))/(2*3.14*math.sqrt(covariance[0]*covariance[1]))

correct = 0
total = 0
for key, value in test_data.iteritems():
	for values in value:
		max_val = 0
		max_violation = ''
		for k, val in models.iteritems():
			"""
			for mean, var in zip(val.means_, val.covariances_):
				temp = gaussian_value(values, mean, [var[0][0],var[1][1]])
				if (temp > max_val):
					max_violation = k
					max_val = temp
			"""
			a = val.predict(values.reshape(1, -1))
			#print val.means_[a], val.covariances_[a][0], val.covariances_[a][0][0][0], val.covariances_[a][0][1][1]
			temp = gaussian_value(values, val.means_[a][0], [val.covariances_[a][0][0][0], val.covariances_[a][0][1][1]]) 
			if (temp > max_val):
					max_violation = k
					max_val = temp

		if max_violation == key:
			correct = correct+1
		total = total + 1
		print total, "of", len(test_data)

print "correct = ", correct, "total = ", total 
print "accuracy = ",
print float(correct)/total
#gmix = fit_samples(np.array(dict['Improper storage trash: res'][:5000]))
#print gmix.means_
#print gmix.covariances_