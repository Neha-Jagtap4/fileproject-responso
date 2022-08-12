class Demo:

	def __init__(self):
		self.A = 10
		self.__B = 20  #private variable (Abstraction)

	def fun(self):
		print("Inside fun")
		self.__gun()   #call function
		
	def __gun(self):
		print("Inside gun")

def main():
	obj = Demo()
	print(obj.A)
	obj.fun()
   	#obj.gun()  private function
	#print(obj.__B)

if __name__ == "__main__":
	main()

# A is the public 
# B is the private