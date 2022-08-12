import java.util.*;

class Demo{
	public int i=10;
	public static int j=20;

	public void fun(){
		System.out.println("Inside function");
		System.out.println(i);  //10
		System.out.println(j);  //20
	}

	public static void gun(){
		System.out.println("Inside gun");
		System.out.println(j);  //20
		//System.out.println(i);
	}
}

class Static{
	public static void main(String arg[]){
		Demo dobj = new Demo();
		dobj.fun();
		dobj.gun();
	
		System.out.println("NOn static function is:" +dobj.i);
		System.out.println("Static function is:" +dobj.j);

	}
}