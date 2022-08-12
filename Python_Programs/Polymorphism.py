#Polymorphism --->Methodoverloading nahi
class Demo:
	def __init__(self):
		self.i=10;
		self.j=20;

	def Add(self,a):
		print("Inside add with one parameter")

	def Add(self,a,b):
		print("Inside add with two parameter")

#If we want to exceute the program ,don't write Function repeadtly ,write once once.

def main():
	obj = Demo()
	obj.Add(10)
#	obj.Add(11,21)

if __name__ =="__main__":
	main()