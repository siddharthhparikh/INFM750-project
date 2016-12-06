
import csv

d = {}
with open('datasets/data_boston-2.csv', 'r') as csvfile:
    csvfile.readline()
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        date = row[1].split(' ')[0]
        if date != '' and int(date.split('/')[2])>2007 and row[12] != '':
            zipcode = int(row[12])
            date = (int(date.split('/')[0]),int(date.split('/')[2]))
            if d.has_key(zipcode):
                if d[zipcode].has_key(date):
                    d[zipcode][date] = d[zipcode][date] + 1
                else:
                    d[zipcode][date] = 1
            else:
                d[zipcode] = {}
                d[zipcode][date] = 1

temp = {}
for key, value in d.iteritems():
    if len(value) == 100:
        temp[key] = value
    
d = temp
del temp

count = 0
for key, value in d.iteritems():
    #print str(key)+',',
    #with open('csv/'+str(key)+'.csv', 'w+') as csvfile:
    average = 0
    for j in range(2008,2017):
        for i in range(1,13):
            count += 1
            if value.has_key((i,j)):
                average += value[(i,j)]
                #csvfile.write(str(value[(i,j)]))
            #csvfile.write('\n')
    print average/100
count = 0
"""
for key, value in d.iteritems():
    if key == 2108:
        for k, v in value.iteritems():
            count += 1
            print key, k, v

print count
"""