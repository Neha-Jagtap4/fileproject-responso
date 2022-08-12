"""
Case Study
	- Euclidean Distance : Calculate the distance between two points
	- User Defined Classifier is used
"""

#importing some required libraries
from sklearn import tree
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def euc(a,b):
	return distance.euclidean(a,b)

class MarvellousKNN():
	def fit(self,TrainingData,TrainingTarget):
		self.TrainingData = TrainingData
		self.TrainingTarget = TrainingTarget

	def predict(self,TestData):
		predictions = []
		for row in TestData:
			lebel = self.closest(row)
			predictions.append(lebel)
		return predictions

	def closest(self,row):
		bestdistance = euc(row,self.TrainingData[0])
		bestindex = 0
		for i in range(1,len(self.TrainingData)):
			dist = euc(row,self.TrainingData[i])
			if dist<bestdistance:
				bestdistance = dist
				bestindex=i
		return self.TrainingTarget[bestindex]


def MarvellousKNeighbor():
	border = "-"*50
	#Step 1 : Load he datasets 
	iris = load_iris()

	data = iris.data
	target = iris.target

	print(border)
	print("Actual data set")
	print(border)

	for i in range(len(iris.target)):
		print("ID : %d,Label %s,Features : %s"%(i,iris.data[i],iris.target[i]))
	print("Size of Training data set %d"%(i+1))

	#Step 2 : Data Training
	data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

	print(border)
	print("Training data set")
	print(border)

	for i in range(len(data_train)):
		print("ID : %d,Label : %s,Features : %s"%(i,data_train[i],target_train[i]))
	print("Size of Training data set %d"%(i+1))

	print(border)
	print("Test data set")
	for i in range(len(data_test)):
		print("ID : %d,Label:%s,Features : %s"%(i,data_test[i],target_test[i]))
	print("Size of test data set %d"%(i+1))
	print(border)

    #User defined Algorithm classifier
	classifier = MarvellousKNN()

	#Step 3 : Train the Algorithm
	classifier.fit(data_train,target_train)
	
	#Step 4 : Data Testing
	predictions = classifier.predict(data_test)
	
	#Step 5 : Calculate Accuracy
	Accuracy = accuracy_score(target_test,predictions)

	return Accuracy

def main():
	Accuracy = MarvellousKNeighbor()
	print("Accuracy of classification algorithm with K Neighbors Classifier is: ",Accuracy*100,"%")

if __name__ == "__main__":
	main()