def Display():
	for i in range(0,4):
		print("Marvellous")

def DisplayX(ino):
	print("By using while loop");
	i=0
	while(i<ino):
		print("Marvellous")
		i=i+1
		DisplayX(ino)  #recurion parat kela tr infinty jato aah ani while la stop kela tr o/p yett aah

def main():
	Display()
	ino = int(input("Enter number: "))
	iret = DisplayX(ino)

if __name__ == "__main__":
	main()