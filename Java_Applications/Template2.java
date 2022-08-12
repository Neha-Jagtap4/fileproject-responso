import java.lang.*;

interface calculator<T>{
	T Display(T ino1,T ino2);
}


class Demo<T>implements calculator<T>{
	public T Display(T ino1, T ino2){
		System.out.println("Inside display");
return ino1;
	}
}

class Template2{
	public static void main(String arg[]){
		Demo<Integer>obj = new Demo<Integer>();
		Integer o = obj.Display(10,20);
		System.out.println(o);
	}
	
}