"""
Case Study
	- Data contains three types of flower : [setosa],[versicolor] and [virginica]
	- Flower can be based on : [sepal-length] ,[sepal-width],[petal-length] and [petal-width]
	- From the given [sepal] and [petal], identify types of flower [setosa],[versicolor] and [virginica]
"""
######################################

#Author : Neha Chandrakant Jagtap
#Date : 10-Feb-2022

#Classifier : RandomForest Tree
#Dataset : Iris set
#Features : sepal and petal
#Label : Iris-setosa, Iris-versicolor and Iris-virginica
#Training dataset : 150 Entries
#Testing dataset : 3 Entry

######################################
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

data = iris.data
target = iris.target

data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.2,random_state=30)
clf = RandomForestClassifier(n_estimators=2,min_samples_split=3,min_samples_leaf=2)

clf.fit(data_train,target_train)

predict_clf = clf.predict(data_test)
Accuarcy = accuracy_score(predict_clf,target_test)
print(Accuarcy)

