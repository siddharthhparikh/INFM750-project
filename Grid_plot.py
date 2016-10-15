import matplotlib.pyplot as plt
import csv 
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

data_list = []
lat_min = 99
lat_max = 0
long_min = 99
long_max = 0

print "data done"

violation_map = {}
i=0
for key,value in data.iteritems():
	random.shuffle(value)
	if len(value) > 19000:
		violation_map[key] = i
		i = i+1
		for val in value[:19000]:
			if val[0] > lat_max:
				lat_max = val[0]
			if val[0] < lat_min:
				lat_min = val[0]
			if val[1] > long_max:
				long_max = val[1]
			if val[1] < long_min:
				long_min = val[1]
			data_list.append(val)


print "data list done"
del data

division = 10000
lat_interval = (lat_max-lat_min)/division
long_interval = (long_max-long_min)/division

count_in_grid = [[[0,0,0,0] for i in range(division)] for j in range(division)]

for i in range(division):
	print i, " of ", division
	for j in range(division):
		for l in data_list:
			if l[0] < lat_min + (i+1)*lat_interval and l[0] > lat_min + i*lat_interval and l[1] < long_min + (i+1)*long_interval and l[1] > long_min + i*long_interval:
				count_in_grid[i][j][violation_map[l[3]]] = count_in_grid[i][j][violation_map[l[3]]] + 1

print count_in_grid