"""
Case Study
	- Supervised Machine Learning based Titanic case study
	- Data contains dependent columns as label [Survived] or not
	- Data identified based on [PassengerId],[Pclass],[Sex],[Age],[SibSp],[Parch],[zero],[Fare],[Embarked] 
	- Survived : [1] and Non-survived : [0]
"""

######################################

#Author : Neha Chandrakant Jagtap
#Date : 08-March-2022

#Classifier : Logistic Regression
#Dataset : Titanic 
#Features : PassengerId,Pclass,Sex,Age,SibSp,Parch,zero,Fare,Embarked
#Label : Survived
#Training dataset : 892 Entries
#Testing dataset : 1 Entry

######################################

#importing some required libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure,show
import seaborn as sns
from seaborn import countplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def MarvellousTitanicPredictor():
	#Step1 : Reading the csv file
	titanic_data = pd.read_csv("MarvellousTitanic.csv")

	print("First 5 enteries from load datasets")
	print(titanic_data.head())
	print(titanic_data.shape)

	#Step2 : Analyze the data
	print("Visualisation : Survived and non - survved passengers")
	ret =figure()
	target = "Survived"
	print(ret)
	print(target)

	ret2 = countplot(data=titanic_data,x=target).set_title("Marvellous : Survived and non survived passengers")	
	print("Countplot:",ret2)
	show()

	print("Visualisation : Survived and non survived based on Gender")
	figure()
	target = "Survived"

	countplot(data = titanic_data,x=target,hue="Sex").set_title("Marvellous : Survived and non survived based on Gender")
	show()

	print("Visualisation : Survived and non survived based on the Passanger class")
	figure()
	target = "Survived"

	countplot(data=titanic_data,x=target,hue="Pclass").set_title("Marvellous : Survived and non survived based on the Passanger class")
	show()

	print("Visualisation : Survived and non survived based on Age")
	figure()
	titanic_data["Age"].plot.hist().set_title("Marvellous : Survived and non survived based on Age")
	show()

	print("Visualisation : Survived and non survived based on Fare")
	figure()
	titanic_data["Fare"].plot.hist().set_title("Marvellous : Survived and non survived based on Fare")
	show()

	#Step3: Data Cleaning
	titanic_data.drop("zero",axis = 1,inplace = True)
	
	print("First 5 enteries from loaded datasets after removing zero column")
	print(titanic_data.head(5))

	print("Values of Sex column")
	print(pd.get_dummies(titanic_data["Sex"]))

	print("Values of Sex column from removing one field")
	Sex = pd.get_dummies(titanic_data["Sex"],drop_first = True)
	print(Sex.head(5))

	print("Values of Pclass column from removing one field")
	Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first = True)
	print(Pclass.head(5))

	print("Values of data set after concatenating new coulmn")
	titanic_data = pd.concat([titanic_data,Sex,Pclass],axis=1)
	print(titanic_data.head(5))

	print("Values of data set after removing irrevelent column")
	titanic_data.drop(["Sex","SibSp","Parch","Embarked"],axis=1,inplace=True)
	print(titanic_data.head(5))

	x = titanic_data.drop("Survived",axis=1)
	y = titanic_data["Survived"]

	#Step 4 : Data Training
	X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.5)

	#Step 5 : Algorithm
	logmodel = LogisticRegression()

	#Step 6 : Train the algorithm
	logmodel.fit(X_train,y_train)

	#Step 7 : Data Testing
	predictions = logmodel.predict(X_test)

	#Step 8 : Calculate Accuracy
	print("Classification report of Logistic Regression")
	print(classification_report(y_test,predictions))

	print("Confusion matrix of Logistic Regression is: ")
	print(confusion_matrix(y_test,predictions))

	print("Accuracy of Logistic Regression is: ")
	print(accuracy_score(y_test,predictions))

def main():
	print("Supervised Machine Learning")
	print("Logistic Regression")
	MarvellousTitanicPredictor()

if __name__ =="__main__":
	main()