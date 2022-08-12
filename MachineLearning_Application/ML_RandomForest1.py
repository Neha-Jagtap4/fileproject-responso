#random forest
import pandas as pd
import numpy as np
##############################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
##############################
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv("bill_authentication.csv")
dataset.head()
print(dataset)
#############################

data = dataset.iloc[:,0:4].values
target = dataset.iloc[:,4].values

data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.2,random_state=0)
sc = StandardScaler()
data_train = sc.fit_transform(data_train)
data_test = sc.fit_transform(data_test)

###############################
classifier = RandomForestClassifier(n_estimators=20,random_state=0)
classifier.fit(data_train,target_train)
predict_clf = classifier.predict(data_test)
####################################
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Confusion matrix: ",confusion_matrix(target_test,predict_clf))
print("Accuracy is: ",accuracy_score(target_test,predict_clf))



