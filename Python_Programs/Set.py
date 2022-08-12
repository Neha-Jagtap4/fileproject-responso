'''1. set : Not sequentail ( we cannot acess index)
   2. set : unordered
   3. list,tuple,array,dictonary : sequentailly it means there indexing at that.
'''

data = {10,20,30,40}
my_list = [10,20,30,40]

print("Datatype is: ",type(data)) 
print("Data length ",len(data))
print("Data from set",data)  # 40 30 20 10  it cannot store data sequentailly in set
print("Data from set",data)
print("Data from list",my_list) # 10 20 30 40 data is sequentally store
print("Datatype is : ",type(my_list))

#print(data[0])

print("Data traversal usinmg loop")
for no in data:
	print(no,end = " ")  # 40 10 20 30  traversal
print(" ")

data1 = {10,20,30,40,20}  #duplicate value are  allowed but store only Once
print(data1)

data2 = {10,20,30,"Hello",True}

print(data2)
#set madhe un sequence asto kit da pn run krr , asha pppppn order yeil