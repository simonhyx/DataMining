import sys
import argparse
import re
import fileinput





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
				dataPoint = [matchobj.group(1),matchobj.group(2),matchobj.group(3),matchobj.group(4),matchobj.group(5),matchobj.group(6),matchobj.group(7),matchobj.group(8),matchobj.group(9),matchobj.group(10),matchobj.group(11)]
				data.append(dataPoint)
				dataLabel.append([matchobj.group(12)])


	return classNames, data, dataLabel
			
				




def main():
	classNames, data, dataLabel =parser()
	print(classNames)
	print(data)
	print(dataLabel)













   
if __name__ == '__main__':
   main ()
   
