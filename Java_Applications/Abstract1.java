import java.util.*;

abstract class Demo{
	public int i,j;
	public Demo(){
		System.out.println("Inside construtor");
		i=10;
		j=20;
	}

	public void fun(){
		System.out.println("Inside function");
	}

	public void gun(){
		System.out.println("Inside gun");
	}

	public abstract void sun();

}

class Hello extends Demo{
	public int x,y;
	public Hello(){
		System.out.println("Inside hello");
		x=10;
		y=20;
	}

	public void fun(){
		System.out.println("Inside hello function");
	}

	public void sun(){
		System.out.println("Inside Hello sun");
	}
}

class Abstract1{
	public static void main(String arg[]){
		//Demo dobj = new Demo();
		
		Demo dobj = new Hello();
		dobj.fun();
		dobj.gun();
		dobj.sun();
		

		System.out.println(dobj.i);
		System.out.println(dobj.j);
		

	}
}