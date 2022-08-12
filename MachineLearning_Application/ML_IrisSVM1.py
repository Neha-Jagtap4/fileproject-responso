"""
Case Study
    - Data contains three types of flower class : [Iris-setosa],[Iris-versicolor] and [Iris-virginica]
    - Flower class can be based on : [sepal-length] ,[sepal-width],[petal-length] and [petal-width]
    - From the given [sepal] and [petal], identify types of flower [setosa],[versicolor] and [virginica]
"""

######################################

#Author : Neha Chandrakant Jagtap
#Date : 05-Feb-2022

#Classifier : Support Vector Machine
#Dataset : Iris set
#Features : sepal and petal
#Label : Iris-setosa, Iris-versicolor and Iris-virginica
#Training dataset : 150 Entries
#Testing dataset : 3 Entry

######################################

#importing some required libraries
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def MarvellousSVMIris(URL):
    colnames=["sepal_length_in_cm", "sepal_width_in_cm","petal_length_in_cm","petal_width_in_cm", "class"]

    #Step 1 : Reading the cscv file from url 
    dataset = pd.read_csv(URL, header = None, names= colnames )
    print("Dataset loaded succesfully.")

    print("Dataset is ")
    print(dataset.head())

    dataset = dataset.replace({"class":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})
    print("After encoding dataset is : ")
    print(dataset.head())
    
    X = dataset.iloc[:,:-1] #we can take the features
    y = dataset.iloc[:, -1].values

    #Step 2 : Data Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    #Step 3 : SVC with linear
    classifier = SVC(kernel = 'linear', random_state = 0)

    #Step 4 : Train the Algorithm
    classifier.fit(X_train, y_train)

    #Step 5 : Data Testing
    y_pred = classifier.predict(X_test)

    #Determine the performance of classification model for a test data
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    #Step 6 : Calculate Accuracy
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

def main():
    MarvellousSVMIris("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

if __name__ == "__main__":
    main()