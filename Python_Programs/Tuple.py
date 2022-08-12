#In tuple data is immutable

data = (10,20,30,40,"Marvellous")

print(type(data))
print("Data is:  ",data)
print("Index at:  ",data[0])
print("Element at index   ",data[1])
print(data[2])
print(data[3])
print(data[4])
print(data[-1])
print(data[-2])

#data[0] = 11
#print(data[0])

data = list(data)  #don't do like this
data.append(50)

print("Upadted data is :  ",data)