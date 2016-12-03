
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
    if len(value) >= 76:
        temp[key] = value
    
d = temp
del temp

for key, value in d.iteritems():
    for k, v in value.iteritems():
        print key, k, v