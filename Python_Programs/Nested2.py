#0X100
def fun():
	print("Inside fun")

#0X200
def gun(func_name):
	print("Inside gun")
	func_name()

def main():
	gun(fun)

if __name__ == "__main__":
	main()