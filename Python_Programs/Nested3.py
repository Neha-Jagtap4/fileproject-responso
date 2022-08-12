def fun():
	print("Inside fun")
	
	def sun():
		print("Inside sun")

	return sun()

def gun(obj):
	print("Inside gun")
	obj()
	
def main():
	gun(fun)
	

if __name__=="__main__":	
	main()