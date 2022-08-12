i=0  #unbound error yeto aah then how to solve
#global kela aah then how write this global variable for recursion

def fun():
	global i  #then write the global inside the function
	if(i<5):
		print(i)
		i=i+1
		fun()  #Recursion call

def main():
	fun()

if __name__ == "__main__":
	main()