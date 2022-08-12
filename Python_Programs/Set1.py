#Program : Accept the numbers from user and search the element where present or not by using set datatype.

print("Enter number of elements")
size = int(input())

data = {0}
print(type(data)) #set

for i in range(size):
	print("Enter element ")  # duplicate are allowed but store once only
	no = int(input())
	data.add(no)
#set is use tosech the data fast
print("Data from set:  ",data)

print("Enter element to search")
value = int(input())

if value in data:
	print("Element is there")
else:
	print("Element is NOT there")

#data = {} ..dict
#data = () ..tuple
#data = [] ..list
#data = set() ..set