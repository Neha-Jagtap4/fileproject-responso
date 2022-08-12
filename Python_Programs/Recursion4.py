def fun():
	i=0
	if(i<5):
		print(i)
		i=i+1
		fun()  #Recursion call
#infinity 

def main():
	fun()



if __name__ == "__main__":
	main()