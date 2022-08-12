import java.util.*;

class Base{
	public void Fun(){
		System.out.println("Base function\n");
	}

	public final void gun(){
		System.out.println("Base gun\n");
	}
}

class Derived extends Base{
	public void Fun(){
		System.out.println("Derived Function\n");
	}

	/*public void gun(){
		System.out.println("Derived gun\n");
	}*/
}

class Final1{
	public static void main(String arg[]){
		Base bobj = new Derived();
		bobj.Fun();
		bobj.gun();

		 
	}
}