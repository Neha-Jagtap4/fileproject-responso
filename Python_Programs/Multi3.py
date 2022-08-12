#morningchiaarti
import os
import multiprocessing

def Fun(X):
	print("Inside Fun")
	print("PID is Fun: ",os.getpid())
	print("PPID of Fun: ",os.getppid())
	for i in range(X):
		print("Fun : ",i)

def Gun(X):
	print("Inside Gun")
	print("PID is Gun: ",os.getpid())
	print("PPID of Gun:",os.getppid() )
	for i in range(X):
		print("Gun : ",i)

def main():
	No=5
	print("Inside main",os.getpid())
	process1 = multiprocessing.Process(target= Fun,args= (No,))
	process2 = multiprocessing.Process(target = Gun,args=(No,))
	
	process1.start()
	process2.start()

	process1.join()
	process2.join()

	print("End of main")

if __name__ =="__main__":
	main()