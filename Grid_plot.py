import matplotlib.pyplot as plt
import csv 
import random
import numpy as np
import math
import matplotlib.patches as patches

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
lat_max = -99
long_min = 99
long_max = -99

print "data done"

violation_map = {}
i=0
for key,value in data.iteritems():
	random.shuffle(value)
	if len(value) > 20000:
		violation_map[key] = i
		i = i+1
		for val in value[:20000]:
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
count = {}
print lat_max, lat_min, long_max, long_min
lat_range_min = 999
lat_range_max = -999
long_range_min = 999
long_range_max = -999

for l in data_list:
	lat_key = int(math.floor((l[0]-lat_min)*1000))
	long_key = int(math.floor(math.fabs(l[1]-long_min)*1000))
	if lat_key > lat_range_max:
		lat_range_max = lat_key
	if lat_key < lat_range_min:
		lat_range_min = lat_key
	if long_key > long_range_max:
		long_range_max = long_key
	if long_key < long_range_min:
		long_range_min = long_key
	
	if not count.has_key((lat_key, long_key)):
		count[(lat_key, long_key)] = [0 for j in range(len(violation_map))]
	count[(lat_key, long_key)][violation_map[l[2]]] = count[(lat_key, long_key)][violation_map[l[2]]] + 1

print lat_range_min, lat_range_max, long_range_min, long_range_max
"""
for key,value in count.iteritems(): 
	print key, value
"""
"""
lat_range_min = int(math.floor((lat_min-math.floor(lat_min))*1000))
lat_range_max = int(math.floor((lat_max-math.floor(lat_max))*1000))
long_range_min = int(math.floor((long_min-math.floor(long_min))*1000))
long_range_max = int(math.floor((long_max-math.floor(long_max))*1000))
"""
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim([lat_range_min, lat_range_max])
ax.set_ylim([long_range_min, long_range_max])

print lat_range_min, lat_range_max, long_range_min, long_range_max
for i in range(lat_range_min, lat_range_max):
	for j in range(long_range_min, long_range_max):
		#print i,j
		if count.has_key((i,j)):
			tot = count[(i,j)][0]+count[(i,j)][1]+count[(i,j)][2]
			red = int(count[(i,j)][0]*255/tot)
			blue = int(count[(i,j)][1]*255/tot)
			green = int(count[(i,j)][2]*255/tot)
			color = '#'+('0'+str(hex(red).split('x')[1]))[-2:] + ('0'+str(hex(blue).split('x')[1]))[-2:] +('0'+str(hex(green).split('x')[1]))[-2:]
			ax.add_patch(
				patches.Rectangle(
					(i, j), 
					1, 
					1,
					facecolor=color,
					linewidth=0,
				)
			)

fig.savefig('rect.png', dpi=1000, bbox_inches='tight')
plt.show()
"""
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.add_patch(
    patches.Rectangle(
        (0.1, 0.1),   # (x,y)
        0.5,          # width
        0.5,          # height
    	facecolor = color,
    )
)
fig1.savefig('rect1.png', dpi=90, bbox_inches='tight')
plt.show()
"""
"""
division = 1000
lat_interval = (lat_max-lat_min)/division
long_interval = (long_max-long_min)/division

count_in_grid = [[[0,0,0,0] for i in range(division)] for j in range(division)]
print "array init done"

for i in range(division):
	print i, " of ", division
	for j in range(division):
		for l in data_list:
			if l[0] < lat_min + (i+1)*lat_interval and l[0] > lat_min + i*lat_interval and l[1] < long_min + (i+1)*long_interval and l[1] > long_min + i*long_interval:
				count_in_grid[i][j][violation_map[l[3]]] = count_in_grid[i][j][violation_map[l[3]]] + 1

print count_in_grid
"""

