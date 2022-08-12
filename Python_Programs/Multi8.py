import os
import multiprocessing

def Square(No):
	print("PID IS: ",os.getpid())
	return No*No

def main():
	data = [5,3,6,8,2]

	p = multiprocessing.Pool()

	result = list()
	#for i in range(len(data)):
	result = p.map(Square,data)
	print("Result is: ",result)

if __name__ == "__main__":
	main()