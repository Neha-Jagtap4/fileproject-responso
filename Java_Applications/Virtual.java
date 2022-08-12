import java.util.*;

class Base{
	public void fun(){
		System.out.println("Base fun");
	}
	public void gun(){
		System.out.println("Base gun");
	}
}

class Derived extends Base{
	public void fun(){
		System.out.println("Derived fun");
	}
	public void sun(){
		System.out.println("Derived sun");
	}
}
class Virtual{
	public static void main(String agr[]){
		Base bobj=new Derived();
		bobj.fun();
		bobj.gun();
		bobj.sun();//error generate krn

	}
}