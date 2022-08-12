import os
import multiprocessing

def Fun(X):
	print("Inside Fun")
	print("PID is Fun: ",os.getpid())
	for i in range(X):
		print("Fun : ",i) 
                                                               
def Gun(X):
	print("Inside Gun")
	print("PID is Gun: ",os.getpid())
	for i in range(X):
		print("Gun : ",i)

def main():
	print("Inside main",os.getpid())
	Fun(5)
	Gun(10)

if __name__ =="__main__":
	main()