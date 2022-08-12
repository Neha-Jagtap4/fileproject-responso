import java.util.*;

final class Base
{
	public int i;
	public void fun(){
		System.out.println("Base function");
	}
}

class Derived {
	public class Base bobj;
	public Derived(){
		class Base bobj = new class Base(bobj);
	}

	public void gun(){
		bobj.fun();
	}

}
class Final3{
	public static void main(String arg[]){
		Base bobj = new Base();
		bobj.fun();
		Derived dobj = new Derived();
		dobj.gun();
	}

}