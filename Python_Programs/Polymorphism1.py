#MethodOverridding  Runtime 

class Base:
	def __init__(self):
		self.i=10
		self.j=20

	def fun(self):
		print("Base funtion")

	def gun(self):
		print("Base gun")

class Derived(Base):
	def __init__(self):
		self.a=11
		self.b=21

	def fun(self):
		print("Derived funtion")

	#def gun(self):
	#	print("Derived gun")

	def sun(self):
		print("Derived sun")

def main():
	bobj = Base()
	dobj = Derived()
	dobj.fun()
	dobj.gun()  #child kade nasel tr parent kadun yeil ani child kade aasel tr tech o/p yeil
	dobj.sun()
	bobj.fun()
	#bobj.sun()

if __name__ == "__main__":
	main()