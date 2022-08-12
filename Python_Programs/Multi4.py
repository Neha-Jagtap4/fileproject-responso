#morningchiaarti
import os
import threading

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
	thread1 = threading.Thread(target= Fun,args= (No,))
	thread2 = threading.Thread(target = Gun,args=(No,))
	
	thread1.start()
	thread2.start()

	thread1.join()
	thread2.join()

	print("End of main")

if __name__ =="__main__":
	main()