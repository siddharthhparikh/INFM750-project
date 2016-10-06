import csv 
import operator
dict = {}
with open('datasets/data_boston.csv', 'r') as csvfile:
	file = csv.reader(csvfile, delimiter=',')
	for row in file:
		if row[5] == 'Improper storage trash: res':
			dict[row[1]] = [row[14], row[15]]

#sorted(timestamps, key=lambda d: map(int, d.split('-')))
sorted_x = sorted(dict.items(), key=operator.itemgetter(0))
#print sorted_x[:20]
#results.sort(key=lambda r: r.person.birthdate)

import matplotlib.pyplot as plt

plt.ion()
plt.xlim([42.2,42.4])
plt.ylim([-71.25,-70.97])
itr = 0
for item in sorted_x:
	#print item[1][0]
	plt.scatter(float(item[1][0]), float(item[1][1]))
	plt.pause(0.01)
	fname = '_tmp%05d.png'%itr
	plt.savefig(fname)
	itr = itr + 1 
	plt.pause(0.01)

import os
os.system("rm movie.mp4")
os.system("ffmpeg -r "+str(10)+" -b 1800 -i _tmp%05d.png movie.mp4")
os.system("rm _tmp*.png")
