"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

#Parameters to set
mu_x = 0
variance_x = 3

mu_y = 0
variance_y = 15

#Create grid and multivariate normal
x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
"""

import csv 
import operator
dict = {}
with open('datasets/data_mgm.csv', 'r') as csvfile:
	file = csv.reader(csvfile, delimiter=',')
	for row in file:
		if dict.has_key(row[11]):
			dict[row[11]].append([row[23], row[24]])
		else:
			dict[row[11]] = [[row[23], row[24]]]


for key, values in dict.iteritems():
	#print key, values
	#break

#sorted_x = sorted(dict.items(), key=operator.itemgetter(0))
