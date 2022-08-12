def Outer():
	print("Inside Outer Function")

	#Inner()  ---->This is not call 
	def Inner():
		print("Inside Inner Function")

		#def Inner1():
		#	print("Inside Inner1 function")

	return Inner
	#OR return Inner()
	#Function is the first class object in python

#Broker .....sarkh
def main():
	func_name = Outer()  #its call outer function
	#func_name = Inner() #its call inner function
	func_name() 

if __name__ == "__main__":
	main()

#function is defining other function
#function is returning another function