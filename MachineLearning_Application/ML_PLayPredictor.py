"""
Case Study
	- Data contains [play]
	- Play-predictor data identified based on [wether] and [Temperature]
	- From the given data [wether] and [Temperature], identified the play [yes or no]
"""

######################################

#Author : Neha Chandrakant Jagtap
#Date : 25-Feb-2022

#Classifier : KNeighbors
#Dataset : Play-predictor
#Features : wether and Temperature
#Label : play
#Training dataset : 41 Entries
#Testing dataset : 1 Entry

######################################

#importing some required libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Step 1 : Reading csv file
dataset = pd.read_csv("MarvellousInfosystems_PlayPredictor.csv")
ret1=dataset.head()
print(ret1)

ret2=dataset.shape
print(ret2)

data = dataset.Wether
target = dataset.Temperature
Label = dataset.Play

print(data)
print(target)
print(Label)

########################################

Wether = ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#######################################

le = LabelEncoder() #Numeric madhe kela aah
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label1 = le.fit_transform(Wether)
label2 = le.fit_transform(temp)
label3 = le.fit_transform(play)
 
# printing label
print(label1)
print(label2)
print(label3)

######################################

features = list(zip(label1,label2)) 
print(features)

######################################
# Train the data
#data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=1/2)


classifier = KNeighborsClassifier(n_neighbors=3)  #Algorithm

#Train the Algorithm
classifier.fit(features,label3)

# Data Testing
predictions = classifier.predict([[0,2]])

print(predictions)

if predictions:
	print("Yes[1]: You can Play")
else:
	print("No[0]: You cannot Play")

#####################################

#Accuracy = accuracy_score(label3,predictions)

#print("Accuracy of classification algorithm with K Neighbors Classifier is: ",Accuracy,"%")