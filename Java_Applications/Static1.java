import java.util.*;

class Demo{
	public int i,j;
	public static int x,y;

	static{
		System.out.println("Inside static block");
		i=10;
		j=20;
	}

	public Demo(){
		System.out.println("INSIDE CONSTRUCTOR");
		i=30;
		j=40;
	}
}

class Static1{
	/*static{
		System.out.println("Inside ht estatic main");
	}*/
	public static void main(String arg[]){
		System.out.println("INSIDE MAIN");
	
		Demo dobj = new Demo();
		Demo dobj1 = new Demo();
	}

}