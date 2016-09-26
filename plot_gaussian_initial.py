
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

"""
#Parameters to set
mu_x = (42.1+42.5)/2
variance_x = 0.0001

mu_y = (-71.3-70.8)/2
variance_y = 0.0001

#Create grid and multivariate normal
x = np.linspace(42.1,42.5,500)
y = np.linspace(-71.3,-70.8,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x-0.1, mu_y-0.1], [[variance_x, 0], [0, variance_y]])
rv1 = multivariate_normal([mu_x+0.1, mu_y+0.1], [[variance_x, 0], [0, variance_y]])
#print rv.pdf([mu_x, mu_y])
#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos)+rv1.pdf(pos),linewidth=0)
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

variance_x = 0.001
variance_y = 0.001
x = np.linspace(38.8,39.4,100)
y = np.linspace(-77.6,-76.8,100)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

s = np.zeros(100)
for key, values in dict.iteritems():
	#print key
	if(key=='Yard'):
		i=0
		for val in values:
			if not (val[1]=='' or val[0]==''):
				print i, "from", len(values)
				print "values:",val[0], val[1]
				rv = multivariate_normal([val[0],val[1]], [[variance_x, 0], [0, variance_y]])
				s = s + rv.pdf(pos)
				print "pdf value:", s
				#print len(rv.pdf(pos))
				i = i+1
		#break
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
