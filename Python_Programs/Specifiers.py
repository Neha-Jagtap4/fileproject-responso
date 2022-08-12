class Demo:

	def __init__(self):
		self.A = 10
		self.__B = 20  #private variable (Abstraction)

	def fun(self):
		print("Inside fun")
		print(self.A)
		print(self.__B)
		self.__gun()
		
	def __gun(self):
		print("Inside gun")
		print(self.A)

def main():
	obj = Demo()
	print(obj.A)
	obj.fun()
	#print(obj.__B)  private

if __name__ == "__main__":
	main()

# A is the public 
# B is the private