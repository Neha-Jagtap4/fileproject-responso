def Outer():
	print("Inside Outer")

	def Inner():
		print("Inside Inner")

		def Inner1():
			print("Inside Inner1")
		return Inner1()

	return Inner
	#return Inner()


def main():
	func_name = Outer()
	func_name()

	#Outer()


if __name__ == "__main__":
	main()