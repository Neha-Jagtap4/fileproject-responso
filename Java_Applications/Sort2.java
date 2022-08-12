import java.util.*;

abstract class ArrayX{
	public int Arr[];
	public ArrayX(int size){
		Arr = sobj.nextInt();
	}

	public void Accept(){
		Scanner sobj = new Scanner(System.in);

		System.out.println("Enter elements");
		for(int i=0; i<Arr.length; i++){
			Arr[i] = sobj.nextInt();
		}
	}
	public void Display(){
		System.out.println("Entered data is:");
		for(int i=0;i<Arr.length; i++){
			System.out.println(Arr[i]);
		}
	}
	public abstract boolean search(int);
}
class Sort1{
	public static void main(String arg[]){
		Scanner  = new Scanner(System.in);
		System.out.println("Enter the size of array");
		int size = sobj.nextInt();
//we cannot create the obj but we can inhreit the abstrat class
		ArrayX obj = new ArrayX(size);  //wrong


	}
}