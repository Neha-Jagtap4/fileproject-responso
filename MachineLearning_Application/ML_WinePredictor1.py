"""
Case Study
	- Data contains [Class] 
	- Data identified based on [Alcohol],[Malic acid],[Ash],[Alcalinity of ash],[Magnesium],[Total phenols],[Flavanoids],[Nonflavanoid phenols],[Proanthocyanins],[Color intensity],[Hue],[OD280/OD315 of diluted wines],[Proline]
"""

######################################

#Author : Neha Chandrakant Jagtap
#Date : 11-Feb-2022

#Classifier : KNeighbors
#Dataset : Wine-predictor
#Features : Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,Total phenols,Flavanoids,Nonflavanoid phenols,Proanthocyanins,Color intensity,Hue,OD280/OD315 of diluted wines,Proline
#Label : Class
#Training dataset : 179 Entries
#Testing dataset : 1 Entry

######################################

#importing some required libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def WinePredictor():
	#Step 1 : Load the dataset
	wine = load_wine()

	print(wine.feature_names)

	print(wine.target_names)

	print(wine.data[0:5])

	print(wine.target)

	#Step 2 : Train the Dataset
	X_train,X_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)

	#Step 3 : Algorithm
	knn=KNeighborsClassifier(n_neighbors=3)

	#Step 4 : Train the Algorithm
	knn.fit(X_train,y_train)

	#Step 5 : Data Testing
	predict_clf=knn.predict(X_test)

	#Step 6 : Calculate Accuracy
	Accuracy = accuracy_score(predict_clf,y_test)

	print("Accuracy is : ",Accuracy*100,"%")

def main():
	WinePredictor()

if __name__ == "__main__":
	main()