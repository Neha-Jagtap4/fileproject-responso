import java.util.*;

class Interface1{
	public static void main(String args[]){
		Marvellous obj = new Marvellous();
		obj.fun();
		obj.fun(11);
		obj.gun();
		System.out.println("Value of hello1 : " +hello1.i);
		System.out.println("Value of hello1 : " +hello2.i);
	}
}

interface hello1{
	int i=10;
	void fun();
	void gun();
}

interface hello2{
	int i = 20;
	void fun(int i);
}

class Marvellous implements hello1, hello2{
	public void fun(){
		System.out.println("Inside function\n");
	}

	public void fun(int i){
		System.out.println("Inside function void fun\n");
	}

	public void gun(){
		System.out.println("Inside gun\n");
	}
}