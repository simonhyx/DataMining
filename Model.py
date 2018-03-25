import sys
import argparse
import re
import fileinput
import os

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn import mixture
from sklearn import cluster
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from itertools import permutations
# reads lines from input file and seperates them into classNames, data, and the label for the data
# all three are returned as lists
# We assume the first line of the file are class names of the data
# We assume the rest of the file is data. 
# data and dataLabel are stored as list of lists
# data and dataLabel contains list of dataPoints, where each dataPoint is stored as a list

from scipy import optimize




def parser():
	test = 1
	classNames = [1,2]
	data = []
	dataLabel = []
	regex1 = re.compile('^(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*)$', re.IGNORECASE)
	regex2 = re.compile('^.*(\d\d):00:00,(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*)$', re.IGNORECASE)
	for line in fileinput.input():
		if test == 1:
			
			matchobj = regex1.match(line)
			if matchobj:
				classNames = [matchobj.group(1),matchobj.group(2),matchobj.group(3),matchobj.group(4),matchobj.group(5),matchobj.group(6),matchobj.group(7),matchobj.group(8),matchobj.group(9),matchobj.group(10),matchobj.group(11),matchobj.group(12)]
				test = 0
				
		else:
			
			matchobj = regex2.match(line)
			if matchobj:
				#print("group = ")
				#print(matchobj.group(1))
				#print(matchobj.group(2))
				dataPoint = [0,int(matchobj.group(2)),int(matchobj.group(3)),float(matchobj.group(4)),float(matchobj.group(5)),float(matchobj.group(6)),float(matchobj.group(7)),float(matchobj.group(8)),float(matchobj.group(9)), int(matchobj.group(1)) ]#int(matchobj.group(10)),int(matchobj.group(11))]
				data.append(dataPoint)
				dataLabel.append([int(matchobj.group(12))])


	return classNames, data, dataLabel
			
				
def plotData(X,Y, title, ):
	plt.plot(X,Y)
	plt.legend()
	plt.title(title)
	plt.xlabel('max depth of regression tree')
	plt.ylabel('R squared score')
	plt.show()
				
def splitByWeather(x,y, dic):
	i = 0
	clear = []
	clear_count = []
	mist = []
	mist_count = []
	light_snow = []
	light_count = []
	heavy_rain = []
	heavy_count = []
	for element in x:
		if i == 1:
			print(element)
		if element[dic["weather"]] == 1:
			clear.append(element)
			clear_count.append(y[i])
		elif element[dic["weather"]] == 2:
			mist.append(element)
			mist_count.append(y[i])
			
		elif element[dic["weather"]] == 3:
			light_snow.append(element)
			light_count.append(y[i])
		elif element[dic["weather"]] == 4:
			heavy_rain.append(element)
			heavy_count.append(y[i])
		i = i + 1
		
	print("clear count, mist count, light snow count, heavy rain count = " + str(len(clear_count)) + ", "+ str(len(mist_count)) + ", "+ str(len(light_count)) + ", "+ str(len(heavy_count)))
	return clear, clear_count, mist, mist_count, light_snow, light_count, heavy_rain, heavy_count

	
	
	
def ridgeReg(X,y):

	X_train, X_test, y_train, y_test = train_test_split(np.array(X)[:,6:], np.array(y), test_size=0.20, random_state=1)
	#print(X_test)
	regr = KernelRidge(alpha=10, kernel = "polynomial", gamma = 0.5)
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)
	
	index = 0
	for i in y_pred:
		#print("ypred = " + str(i) + " y test = " + str(y_test[index]))
		index = index + 1
		
	#print('Coefficients: \n', regr.coef_)
	# The mean squared error
	print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % r2_score(y_test, y_pred))

	#What were the real predictions?
	y_pred_train = regr.predict(X_train)
	print("Mean squared error on the training set: %.2f"%mean_squared_error(y_train, y_pred_train))
	print("Mean squared error on the test set:     %.2f"%mean_squared_error(y_test, y_pred))
	print("size of X = ", str(len(y)))
	
	
	
	
	
	
	
	
	
def reduction(x,y):
	X = [a[-1] for a in x]
	Y = [a[0] for a in y]
	return X,Y
	
	
	
	
def adaGeneration(x,y, title):
	train_err = []
	test_err = []
	score = []
	for dep in range(1,13):
		a,b,c = adaRegress(x,y,dep)
		train_err.append(a)
		test_err.append(b)
		score.append(c)
		
	axis = [1,2,3,4,5,6,7,8,9,10,11,12]
	plotData (axis, score, title)
	
	
def adaRegress(x,y,dep):
# Fit regression model
	

	rng = np.random.RandomState(1)

	X_train, X_test, y_train, y_test = train_test_split(np.array(x)[:,2:], np.array(y), test_size=0.30, random_state=1)
	regr_1 = DecisionTreeRegressor(max_depth=dep)

	regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=dep),
                          n_estimators=500, random_state=rng)

	regr_1.fit(X_train, y_train)
	regr_2.fit(X_train, y_train)

	
	
	y_1 = regr_1.predict(X_test)
	y_2 = regr_2.predict(X_test)
	y_3 = regr_2.predict(X_train)
	
	"""
	print("Mean squared error: %.2f" % mean_squared_error(y_train, y_3))
	print("Mean squared error: %.2f" % mean_squared_error(y_test, y_1))
	print("Mean squared error: %.2f" % mean_squared_error(y_test, y_2))
	print('Variance score: %.2f' % r2_score(y_test, y_1))
	"""
	
	
	return mean_squared_error(y_train, y_3), mean_squared_error(y_test, y_2), r2_score(y_test, y_2)

	
	
	
	
	
def execute():
	
	
	print (os.path.dirname(os.path.abspath(sys.argv[0])))
	#Mydata = datasets.load_files('data',shuffle='False')
	#X_train, X_test, y_train, y_test = train_test_split(Mydata.data, Mydata.target, test_size=0.20, random_state=0)
	classNames, classifiers, dataLabel =parser()
	print(classNames)
	dic = {}
	
	#gaussRegress(classifiers,dataLabel)
	print("...............")
	adaRegress(classifiers, dataLabel)
	
	for i in range(0,len(classNames)):
		dic[classNames[i]] = i
	
	holiday = []
	holiday_count = []
	weekDay = []
	weekDay_count = []
	weekEnd = []
	weekEnd_count = []
	i = 0
	for x in classifiers:
		#print(x[dic["holiday"]])
		if x[dic["holiday"]] == 1 and x[dic["workingday"]] == 1:
			holiday.append(x)
			holiday_count.append(dataLabel[i])
		elif x[dic["holiday"]] == 1 :
			weekEnd.append(x)
			weekEnd_count.append(dataLabel[i])
		else:
			weekDay.append(x)
			weekDay_count.append(dataLabel[i])
		i = i +1

	#print(holiday)
	print("holidays")
	adaRegress(weekDay,weekDay_count)
	adaRegress(weekEnd, weekEnd_count)
	hugeList = [holiday,holiday_count,weekDay,weekDay_count,weekEnd,weekEnd_count]
	i=0
	for k in range(0,3):
		x1,y1,x2,y2,x3,y3,x4,y4 = splitByWeather(hugeList[i],hugeList[i+1], dic)
		if len(x1) > 1:

			print("cluster1")
			adaRegress(x1,y1)
			ridgeReg(x1, y1)
			
			#gaussRegress(x1,y1)
		if len(x2) > 1:

			print("cluster2")
			adaRegress(x1,y1)
			ridgeReg(x2,y2)
			#gaussRegress(x2,y2)
		if len(x3) > 1:

			print("cluster3")
			adaRegress(x1,y1)

			ridgeReg(x3,y3)
		if len(x4)>1:

			print("cluster4")
			adaRegress(x1,y1)
			ridgeReg(x4,y4)

		i = i + 2
		
	
	
	
def main():

	classNames, classifiers, dataLabel =parser()
	print(classNames)
	dic = {}
	
	#gaussRegress(classifiers,dataLabel)
	print("...............")
	adaGeneration(classifiers, dataLabel, "Performance of adaBoost over all days")
	
	for i in range(0,len(classNames)):
		dic[classNames[i]] = i
	
	holiday = []
	holiday_count = []
	weekDay = []
	weekDay_count = []
	weekEnd = []
	weekEnd_count = []
	i = 0
	for x in classifiers:
		#print(x[dic["holiday"]])
		if x[dic["holiday"]] == 1 and x[dic["workingday"]] == 1:
			holiday.append(x)
			holiday_count.append(dataLabel[i])
		elif x[dic["holiday"]] == 1 :
			weekEnd.append(x)
			weekEnd_count.append(dataLabel[i])
		else:
			weekDay.append(x)
			weekDay_count.append(dataLabel[i])
		i = i +1

	#print(holiday)
	print("holidays")
	adaGeneration(weekDay,weekDay_count, "ada boost on weekDay data")
	adaGeneration(weekEnd, weekEnd_count, "ada boost on weekEnd data")
	

		
		
		
	"""
	print(np.array(X_train).shape)
	print(np.array(y_train).shape)

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from sklearn import preprocessing
	#plt.scatter(np.array(X_train), np.array(y_train), s=500)
	X_train = np.array(X_train)
	X_train = preprocessing.normalize(X_train, norm='l2')
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X_train[:,4],X_train[:,5],X_train[:,7])
	ax.set_xlabel('Petal width')
	ax.set_ylabel('Sepal length')
	ax.set_zlabel('Petal length')
	ax.set_title("kobe")
	plt.show()

   """
if __name__ == '__main__':
   main ()
   
