import java.util.*;
class Demo{
	public void Display(){
		for(int i=0; i<=10; i++){
			System.out.println("Demo is: " +i);
		}
	}
}


class Hello{
	public void fun(){
		for(int i=0; i<=10; i++){
			System.out.println("Hello is :"+i);
		}
	}
}
class Thread{  
	public static void main(String arg[]){
		Demo dobj = new Demo();
		Hello hobj = new Hello();
		dobj.Display();
		hobj.fun();
	}
}