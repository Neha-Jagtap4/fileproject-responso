import java.util.*;

class Demo{
	public final int i;
	public  int j;

	public Demo(){
		i=10;
		j=11;
		i = i+1;
	}
}

class Final{
	public static void main (String arg[]){
		Demo dobj = new Demo();
		System.out.println("VAlue of j: " +dobj.i);
	}
}
