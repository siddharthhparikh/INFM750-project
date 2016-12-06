
# coding: utf-8

# Long Short-Term Memory network is a type of Recurrent Neural Network

# In[38]:

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[39]:

# fix random seed for reproducibility
numpy.random.seed(7)


# In[40]:
#zipcodes = [2215, 2108, 2109, 2111, 2113, 2114, 2115, 2116, 2118, 2119, 2121, 2122, 2124, 2125, 2126, 2127, 2128, 2130, 2131, 2132, 2134, 2135, 2136]
zipcodes = [2122]
fp = open('results.txt', 'w+')

for ratio in range(60,80,30):
	for lookback in range(1,2,3):
		for neurons in range(10000,10001,3):
			for zipcode in zipcodes:
				# load the dataset
				fp.write("For neurons:"+str(neurons)+" lookback:"+str(lookback)+" zipcode:"+str(zipcode)+" Training size:"+str(ratio)+"\n")
				dataframe = pandas.read_csv('csv/'+str(zipcode)+'.csv', usecols=[0], engine='python')
				dataset = dataframe.values
				dataset = dataset.astype('float32')
				#print "length of dataset = ", len(dataset)

				# In[41]:

				#dataframe.describe()


				# In[42]:

				# normalize the dataset
				scaler = MinMaxScaler(feature_range=(0, 1))
				dataset = scaler.fit_transform(dataset)
				#print len(dataset)


				# In[142]:

				# split into train and test sets - train set ends on May 12
				train_size = ratio
				test_size = len(dataset) - train_size
				train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
				#print(len(train), len(test))


				# In[143]:

				def create_dataset(dataset, look_back=1):
				    dataX, dataY = [], []
				    for i in range(len(dataset)-look_back-1):
				        a = dataset[i:(i+look_back), 0]
				        dataX.append(a)
				        dataY.append(dataset[i + look_back, 0])
				    return numpy.array(dataX), numpy.array(dataY)


				# In[152]:

				# reshape into X=t and Y=t+1
				look_back = lookback
				trainX, trainY = create_dataset(train, look_back)
				testX, testY = create_dataset(test, look_back)


				# In[ ]:

				# reshape input to be [samples, time steps, features]
				trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
				testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


				# In[ ]:

				# create and fit the LSTM network
				model = Sequential()
				model.add(LSTM(neurons, input_dim=look_back))
				model.add(Dense(1))
				model.compile(loss='mean_squared_error', optimizer='adam')
				model.fit(trainX, trainY, nb_epoch=100)


				# In[ ]:
				# make predictions
				trainPredict = model.predict(trainX)
				testPredict = model.predict(testX)
				# invert predictions
				trainPredict = scaler.inverse_transform(trainPredict)
				trainY = scaler.inverse_transform([trainY])
				testPredict = scaler.inverse_transform(testPredict)
				testY = scaler.inverse_transform([testY])
				# calculate root mean squared error
				trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
				fp.write('Train Score: %.2f RMSE\n' % (trainScore))
				testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
				fp.write('Test Score: %.2f RMSE\n' % (testScore))
		
			

				# In[ ]:

				# shift train predictions for plotting
				trainPredictPlot = numpy.empty_like(dataset)
				trainPredictPlot[:, :] = numpy.nan
				trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
				# shift test predictions for plotting
				testPredictPlot = numpy.empty_like(dataset)
				testPredictPlot[:, :] = numpy.nan
				testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
				# plot baseline and predictions
				plt.plot(scaler.inverse_transform(dataset))
				plt.plot(trainPredictPlot)
				plt.plot(testPredictPlot)

				plt.show()
				# lookback = 1, hidden neurons = 4
				# Train Score: 398.36 RMSE
				# Test Score: 314.96 RMSE
				# 
				# lookback = 3, hidden neurons = 4
				# Train Score: 387.69 RMSE
				# Test Score: 214.87 RMSE
				# 
				# lookback = 7, hidden neurons = 4
				# Train Score: 357.99 RMSE
				# Test Score: 146.85 RMSE
				# 
				# lookback = 10, hidden neurons = 8
				# Train Score: 366.99 RMSE
				# Test Score: 149.24 RMSE
				# 
				# 

				# In[ ]:

fp.close()

