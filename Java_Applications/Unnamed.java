import java.util.*;

class Demo{
	public int i,j;
	public static int x,y;

	static{
		x=10;
		y=20;
	}

	public Demo(){
		System.out.println("Inside default constructor");
	}

	public Demo(int a,int b){
		System.out.println("Inside Paramertized constructor");
	}
}
class Unnamed{
	public static void main(String arg[]){
		Demo dobj1 = new Demo();
		Demo dobj2 = new Demo(1,2);

		System.out.println("The value of static block is:" +dobj1.x);
		System.out.println("The value of static block is:" +dobj1.y);
	}
}