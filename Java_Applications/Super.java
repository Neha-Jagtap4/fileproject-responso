import java.util.*;
class Base{
	int x,y;
	public Base(int a,int b){
		System.out.println("Base para Constructor");
		this.x=a;
		this.y=b;
	}

	public void fun(){
		System.out.println("Inside function");
	}
}

class Derived extends Base{
	public int a, b;
	public Derived(){
		super(11,21);
		System.out.println("Derived base paramertized");
	}

	public void gun(){
		System.out.println("Inside gun 1" +super.x);
		super.fun();
		
	}
}

class Super{
	public static void main(String str[]){
		Base bobj=new Base(10,20);
		bobj.fun();

		Derived dobj=new Derived();
		dobj.gun();

	}

	
}