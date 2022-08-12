"""
Case Study
	- Given data is Dependent and Independent columns
	- Dependent data contain [Gender]
	- HeadBrain data identified based on [Age Range],[Head Size(cm^3)],[Brain Weight(grams)]
	- From the given [Age Range],[Head Size(cm^3)],[Brain Weight(grams)],identified [Gender]
"""

######################################

#Author : Neha Chandrakant Jagtap
#Date : 19-March-2022

#Classifier : Linear Regression
#Dataset : Head-Brain
#Features : Age Range,Head Size(cm^3),Brain Weight(grams)
#Label : Gender
#Training dataset : 238 Entries
#Testing dataset : 1 Entry

######################################

#importing some required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def MarvellousHeadBrainPredictor():
	#Step 1 : Reading csv file
	data = pd.read_csv("MarvellousHeadBrain.csv")

	print("Size of data: ",data.shape)
	print(data.head())

	X = data['Head Size(cm^3)'].values
	Y = data['Brain Weight(grams)'].values
	
	#Reshaping the dimension on axis
	X = X.reshape((-1,1))

	n = len(X)

	#Step 2 : Algorithm
	reg = LinearRegression()

	#Step 3 : Train the Algorithm
	reg = reg.fit(X,Y)

	#Step 4 : Data Testing
	y_pred = reg.predict(X)

	r2 = reg.score(X,Y)

	print(r2)

def main():
	print("Supervised Machine Learning")

	print("Linear Regression on head and brain size dataset")
	MarvellousHeadBrainPredictor()

if __name__ == "__main__":
	main()