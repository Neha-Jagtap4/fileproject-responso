import java.util.*;

class FileHandling1{
	public static void main(String arg[]){
		Scanner sobj = new Scanner(System.in);
		int ino1=0,ino2=0;
		float div = 0.0f;

		System.out.println("Enter first number");
		ino1 = sobj.nextInt();

		System.out.println("Enter second number");
		ino2= sobj.nextInt();

try{
	div =  ino1/ino2;
	System.out.println("Division is :" +div);
}

catch(ArithmeticException obj){
	System.out.println("Exception occured: "+obj);
}

catch(Exception iobj){
	System.out.println("Exception is occured in iobj: "+iobj);
}

finally{
	System.out.println("Terminated application");
}
}
}