import java.util.*;

class GTemplate<T>{
	private T t;
	public void  fun(T t) {
		this.t = t;
	
	}
	public T get(){
		return t;
	}
}

public class Template{
	public static void main (String arg[]){

		GTemplate<Integer>integerbox = new GTemplate<Integer>();
		GTemplate<String>stringbox = new GTemplate<String>();

		integerbox.add(new Integer(20));
		stringbox.add(new String("Hello world"));
		System.out.println("Inteder is  " +integerbox.get());
		System.out.println("Integer is :"+stringbox.get());
	}
}