import java.util.*;

class Implements3{
	public static void main(String arg[]){
		Marvellous mobj = new Marvellous();
		mobj.fun();
		mobj.gun();
	}
}

interface Derived{
	void fun();
}

interface Base extends Derived{
	void gun();
}

class Marvellous implements Base{
	public void fun(){
		System.out.println("Inside fun");
	}

	public void gun(){
		System.out.println("Inside gun");
	}
}