"""
Case Study
	- Data contains three types of flower class : [Iris-setosa],[Iris-versicolor] and [Iris-virginica]
	- Flower class can be based on : [sepal-length] ,[sepal-width],[petal-length] and [petal-width]
	- From the given [sepal] and [petal], identify types of flower [setosa],[versicolor] and [virginica]
"""

######################################

#Author : Neha Chandrakant Jagtap
#Date : 01-Feb-2022

#Classifier : Decision Tree
#Dataset : Iris set
#Features : sepal and petal
#Label : Iris-setosa, Iris-versicolor and Iris-virginica
#Training dataset : 150 Entries
#Testing dataset : 3 Entry

######################################

#importing some required libraries
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

#Step 1 : Loading datasets
iris = load_iris()

print("Features name of iris datasets")
#printing the features name of iris data
print(iris.feature_names) 

print("Target names of iris datasets")
#printing the target name of iris data
print(iris.target_names) 

for i in range(len(iris.target)):
	print("ID : %d,Features %s,Label: %s"%(i,iris.data[i],iris.target[i]))

#Step 2 : Loading the data(entries)
test_index = [1,51,101]  
print(test_index)

#Step 3 : Delete the specific [test_index] from dataset
train_target = np.delete(iris.target,test_index) 
print(train_target)
train_data = np.delete(iris.data,test_index,axis=0) 
print(train_data)

test_target = iris.target[test_index]  
print(test_target)
test_data = iris.data[test_index] 
print(test_data) 

#Step 4 : Algorithm
classifier = tree.DecisionTreeClassifier() 

#Step 5 : Train the Algorithm
# [fit] method is use to train algorithm
classifier.fit(train_data,train_target)  
             
print("Values that we removed for testing")
print(test_target) 
print(test_data)

#Step 6 : Data Testing
print("Result of testing")
print(classifier.predict(test_data)) 