"""
Case Study
	- In this, Decision and Knn Algorithm is use to calculate Accuracy
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
#Testing dataset : 4 Entry

######################################

#importing some required libraries
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def MarvellousCalculateAccuracyDecisionTree():
	#Step 1 : Load the datasets
	iris = load_iris()

	data = iris.data
	target = iris.target

	#Step 2 : Data Training
	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

	#Step 3 : Algorithm
	classifier = tree.DecisionTreeClassifier()

	#Step 4 : Train the Algorithm
	classifier.fit(data_train,target_train)

	#Step 5 : Data Testing
	predictions = classifier.predict(data_test)

	#Step 6 : Calculate accuracy
	Accuracy = accuracy_score(target_test,predictions)
	
	return Accuracy

def MarvellousCalculateAccuracyKNeighbor():
	#Step 1 : Load the datasets
	iris = load_iris()

	data = iris.data
	target = iris.target

	#Step 2 : Data Training
	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

	#Step 3 : Algorithm
	classifier = KNeighborsClassifier()

	#Step 4 : Train the Algorithm
	classifier.fit(data_train,target_train)

	#Step 5 : Data Testing
	predictions = classifier.predict(data_test)

	#Step 6 : Calculate accuracy
	Accuracy = accuracy_score(target_test,predictions)

	return Accuracy

def main():
	Accuracy = MarvellousCalculateAccuracyDecisionTree()
	print("Accuracy of classification algorithm with Decision Tree Classifier is: ",Accuracy*100,"%")

	Accuracy = MarvellousCalculateAccuracyKNeighbor()
	print("Accuracy of classification algorithm with K Neighbors Classifier is: ",Accuracy*100,"%")

if __name__ == "__main__":
	main()