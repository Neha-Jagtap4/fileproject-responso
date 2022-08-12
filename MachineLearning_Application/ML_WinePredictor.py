"""
Case Study
	- Data contains [Class] 
	- Data identified based on [Alcohol],[Malic acid],[Ash],[Alcalinity of ash],[Magnesium],[Total phenols],[Flavanoids],[Nonflavanoid phenols],[Proanthocyanins],[Color intensity],[Hue],[OD280/OD315 of diluted wines],[Proline]
"""

######################################

#Author : Neha Chandrakant Jagtap
#Date : 16-Feb-2022

#Classifier : KNeighbors
#Dataset : Wine-predictor
#Features : Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,Total phenols,Flavanoids,Nonflavanoid phenols,Proanthocyanins,Color intensity,Hue,OD280/OD315 of diluted wines,Proline
#Label : Class
#Training dataset : 179 Entries
#Testing dataset : 1 Entry

######################################

#importing some required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Step 1 : Reading csv file
dataset = pd.read_csv("WinePredictor.csv")

iret1 = dataset.head()
print(iret1)

iret2 = dataset.shape
print(iret2)

features1 = ['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
print("Names of features1 : ",features1)

X = dataset[features1]
print("features1 :",X)
y = dataset.Class
print("Class : ",y)

#Step 2 : Train the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/2)

#Step 3 : Algorithm
knn = KNeighborsClassifier()

#Step 4 : Train the Algorithm
knn.fit(X_train,y_train)

#Step 5 : Data Testing
predict_clf = knn.predict(X_test)

print(predict_clf)

#Step 6 : Calculate Accuracy
Accuarcy = accuracy_score(y_test,predict_clf)

print("Accuracy is: ",Accuarcy*100 ,"%")