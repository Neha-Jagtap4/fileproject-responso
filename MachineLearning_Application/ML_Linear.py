import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1,2,3,4,5]).reshape((-1,1))
y = np.array([3,4,2,4,5])

#[1,2,3,4,5]
#[[
#	1,
#	2,
#	3,
#	4,
#	5]]

model = LinearRegression()

model.fit(x,y)

r_sq = model.score(x,y)
print("Coefficient determinant of R Square is : ",r_sq)

print("Intercept (C): ",model.intercept_)

print("Slope (m) : ",model.coef_)

y_predict = model.predict(x)
print("Predicted response : ",y_predict,sep="\n")

y_predict = model.intercept_ + model.coef_ * x
print("Predicted response : ",y_predict,sep="\n")