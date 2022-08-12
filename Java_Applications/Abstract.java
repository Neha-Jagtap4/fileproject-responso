import java.util.*;

Abstract class Demo{
	public int i,j;
	public Demo(){
		System.out.println("Inside construtor");
		//i=10;
		//j=20;
	}

	public void fun(){
		System.out.println("Inside function");
	}

	public void gun(){
		System.out.println("Inside gun");
	}

	/*public abstract void sun(){
		System.out.println("inside sun");
	}*/
}

class Hello extends Demo{
	public int x,y;
	public Hello(){
		System.out.println("Inside hello");
		//x=10;
		//y=20;
	}

	public void fun(){
		System.out.println("Inside hello function");
	}

	public void sun(){
		System.out.println("Inside Hello sun");
	}
}

class Abstract{
	public static void main(String arg[]){
		//Demo dobj = new Demo();
		Demo dobj ;
		dobj = new Hello();
		dobj.fun();
		dobj.gun();
		dobj.sun();
		//dobj1.fun();

		System.out.println(dobj.i);
		System.out.println(dobj.j);
		//System.out.println("The value of obj demo" +dobj.x);
		//System.out.println("The value of obj demo" +dobj.y);


	}
}