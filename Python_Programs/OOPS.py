#Encapsulation--->Arithemtic(class)===>Charactertics  &  Behavior
class Arithmetic:
	def __init__(self,a,b): #Constructor
		print("Inside init(Constructor)")
		self.ino1 = a  #Characteritics
		self.ino2 = b  #Characteritics

	def Addition(self): #self->0X100
		ans = self.ino1 + self.ino2  #Behavior
		return ans

def main():
	print("Enter first number: ")
	x = int(input())

	print("Enter second number: ")
	y = int(input())

	obj = Arithmetic(x,y)
	iret = obj.Addition() # Addition(obj1) Addition(0X100)

	print("Addition is: ",iret)

if __name__ == "__main__":
	main()

#characterticys (ino1,ino2)
#1)instance variable (e.g.PLate madhe sagle vattii... varan-bhat,poli)
#2)class variable(e.g. Loncha)