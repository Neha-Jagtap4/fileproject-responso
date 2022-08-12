import java.util.*;

class Implements1{
	public static void main(String args[]){
		Implements1 obj = new Implements1();
		obj.fun();
		obj.fun(11);
		obj.gun();
		System.out.println("Value of hello1 : "+hello1);
		System.out.println("Value of hello1 : "+hello2.i);
	}
}

interface hello1{
	void fun();
	void gun();
}

interface hello2{
	int i = 20;
	void fun(int i);
}

class Implements1 implements hello1, hello2{
	public void fun(){
		System.out.println("Inside function\n");
	}

	public void fun(){
		System.out.println("Inside function void fun\n");
	}

	public void gun(){
		System.out.println("Inside gun\n");
	}
}