def fun(i):
	#def fun(i=0):
	if(i<5):
		print(i)
		i=i+1
		fun(i)  #Recursion call

def main():
	fun(0)

if __name__ == "__main__":
	main(0)