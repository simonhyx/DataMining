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

from itertools import permutations
# reads lines from input file and seperates them into classNames, data, and the label for the data
# all three are returned as lists
# We assume the first line of the file are class names of the data
# We assume the rest of the file is data. 
# data and dataLabel are stored as list of lists
# data and dataLabel contains list of dataPoints, where each dataPoint is stored as a list
def parser():
	test = 1
	classNames = [1,2]
	data = []
	dataLabel = []
	regex1 = re.compile('^(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*),(\w*)$', re.IGNORECASE)
	regex2 = re.compile('^(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*)$', re.IGNORECASE)
	for line in fileinput.input():
		if test == 1:
			
			matchobj = regex1.match(line)
			if matchobj:
				classNames = [matchobj.group(1),matchobj.group(2),matchobj.group(3),matchobj.group(4),matchobj.group(5),matchobj.group(6),matchobj.group(7),matchobj.group(8),matchobj.group(9),matchobj.group(10),matchobj.group(11),matchobj.group(12)]
				test = 0
				
		else:
			
			matchobj = regex2.match(line)
			if matchobj:
				dataPoint = [0,0,int(matchobj.group(3)),float(matchobj.group(4)),float(matchobj.group(5)),float(matchobj.group(6)),float(matchobj.group(7)),float(matchobj.group(8)),float(matchobj.group(9)),matchobj.group(10),matchobj.group(11)]
				data.append(dataPoint)
				dataLabel.append([matchobj.group(12)])


	return classNames, data, dataLabel
			
				

				
				
				
def plot_clusters(X, c,title="Iris Dataset"):
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X[:,3],X[:,4],X[:,6],c=[0]*X.shape[0], edgecolor='k')
	ax.set_xlabel('Petal width')
	ax.set_ylabel('Sepal length')
	ax.set_zlabel('Petal length')
	ax.set_title(title)
	plt.show()
    

def find_perm(y1, y2):
	num_classes = y2.max()+1 #add one because 0 is a class
	M = np.zeros((num_classes,num_classes))
	for i1, i2 in zip(y1,y2):
		M[i1][i2] += 1
    # iterate through every possible permutation, careful
    # this will get very slow with a large number of classes!
	best_perm  = range(num_classes)
	best_total = 0
	for perm in permutations(range(num_classes)):
		total = 0
		for i in range(num_classes):
			total += M[i][perm[i]]
		if total>=best_total:
			best_total = total
			best_perm  = perm
	return best_perm


	
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
		
	return clear, clear_count, mist, mist_count, light_snow, light_count, heavy_rain, heavy_count


def makeCluster(x,y):
	print(x)
	X_train, X_test, y_train, y_test = train_test_split(np.array(x)[:,6:], np.array(y), test_size=0.20, random_state=1)
	#print(X_train)
	#plot_clusters(np.array(X_train),np.transpose(np.array(y_train)),title="Raw Data")
	
	GMix = mixture.GaussianMixture(n_components=5)
	GMix.fit(X_train)
	G_pred = GMix.predict(X_test)
	p = G_pred
	y_pred = list(map(lambda x: p[x],G_pred))

	import matplotlib.pyplot as plt
	plt.scatter(np.array(G_pred[:20]), np.array(y_test[:20]), s=500)
	plt.show()
	
	
def main():
	print (os.path.dirname(os.path.abspath(sys.argv[0])))
	#Mydata = datasets.load_files('data',shuffle='False')
	#X_train, X_test, y_train, y_test = train_test_split(Mydata.data, Mydata.target, test_size=0.20, random_state=0)
	classNames, classifiers, dataLabel =parser()
	print(classNames)
	dic = {}
	
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
		print(x[dic["holiday"]])
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

	print(holiday)
	hugeList = [weekDay,weekDay_count,weekEnd,weekDay_count]
	i=0
	for k in range(0,2):
		x1,y1,x2,y2,x3,y3,x4,y4 = splitByWeather(hugeList[i],hugeList[i+1], dic)
		if len(x1) > 0:
			makeCluster(x1,y1)
		if len(x2) > 0:
			makeCluster(x2,y2)
		if len(x3) > 0:
			makeCluster(x3,y3)
		if len(x4)>0:
			makeCluster(x4,y4)
		i = i + 2
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
   
