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

#Classifier : User-Defined 
#Dataset : Head-Brain
#Features : Age Range,Head Size(cm^3),Brain Weight(grams)
#Label : Gender
#Training dataset : 238 Entries
#Testing dataset : 1 Entry

######################################

#importing some required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MarvellousHeadBrainPredictor():
	#Step 1 : Reading csv file
	dataset = pd.read_csv("MarvellousHeadBrain.csv")
	
	ret1 = dataset.head()
	print(ret1)
	ret2 = dataset.shape
	print(ret2)

	print("Size of data set : ",dataset.shape)
	print(dataset.head())
	print(dataset.Gender)
	X = dataset['Head Size(cm^3)'].values
	Y = dataset['Brain Weight(grams)'].values

	mean_x = np.mean(X)
	mean_y = np.mean(Y)

	n = len(X)

	numerator = 0
	denomenator = 10
	for i in range(n):
		numerator+=(X[i] - mean_x)*(Y[i]-mean_y)
		denomenator += (X[i]- mean_x)**2

	m = numerator/denomenator
	c = mean_y- (m*mean_x)

	print("Slope of Regression line is: ",m)
	print("Y Intercept of Regression line is: ",c)

	max_x = np.max(X)+100
	min_x = np.min(X)+100

	x = np.linspace(min_x,max_x,n)

	y = c + m*x

	plt.plot(x,y,color ='#58b970',label = 'Regression line')
	plt.scatter(X,Y,color='#ef5423',label='scatter plot')
	plt.xlabel("Head size in cm3")
	plt.ylabel("Brain weight in grams")

	plt.legend()
	plt.show()

	ss_t = 0
	ss_r = 0

	for i in range(n):
		y_pred = c + m*X[i]
		ss_t += (Y[i] - mean_y) ** 2
		ss_r += (Y[i] - y_pred) ** 2

	r2 = 1 - (ss_r/ss_t)
	print(r2)

def main():
	print("Supervised Machine Learning")

	print("Linear Regression on Head and Brain size of data set")
	MarvellousHeadBrainPredictor()

if __name__ == "__main__":
	main()