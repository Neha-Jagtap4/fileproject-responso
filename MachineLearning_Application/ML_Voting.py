"""
Case Study
	- In this Voting Classifier Algorithm is used
	- Calculate the accuracy by using algorithm
	- Data contains three types of flower : [iris-setosa],[iris-versicolor] and [iris-virginica]
	- Data identified based on : [sepal] and [petal]
"""

######################################

#Author : Neha Chandrakant Jagtap
#Date : 10-March-2022

#Classifier : VotingClassifier
#Dataset : Iris 
#Features : sepal and petal
#Label : Iris-setosa, Iris-versicolor and Iris-virginica

######################################

#importing some required libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

def VotingClassifierr():
	#Step 1 : Load the dataset
	iris = load_iris()

	x = iris["data"]
	y = iris["target"]

	#Step 2 : Train the dataset
	X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.85)

	#Step 3 : Algorithms
	log_clf = LogisticRegression()
	rand_clf = RandomForestClassifier()
	knn_clf = KNeighborsClassifier()

	vot_clf = VotingClassifier(estimators=[('lr',log_clf),('rmd',rand_clf),('knn',knn_clf)],voting="hard")

	#Step 4 : Train the algorithm
	vot_clf.fit(X_train,y_train)

	#Step 5 : Data Testing
	pred = vot_clf.predict(X_test)

	#Step 6 : Calculate accuracy
	print("Testing Accuracy : ",accuracy_score(y_test,pred)*100)

def main():
	print("-----------Voting Classifier----------")
	VotingClassifierr()

if __name__ == "__main__":
	main()