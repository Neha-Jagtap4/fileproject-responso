import java.util.*;

class Demo{  //logic
	public float Division(int x,int y)throws ArithmeticException{ //function thows
		float ians = 0.0f;
		ians = x/y;
		return ians;
	}
}

class FileThrows{  
	public static void main(String arg[]){
		  Scanner sobj = new Scanner(System.in);
		  int ino1=0,ino2=0;
		  float result = 0.0f;
		  Demo dobj = new Demo();  //demo access

		  System.out.println("Enter first number");
		  ino1 = sobj.nextInt();

		  System.out.println("Enter second number");
		  ino2 = sobj.nextInt();

	

	try{
		result = dobj.Division(ino1,ino2);  //function call....dobj
		System.out.println("Division is: " +result);
		
	}

	catch(ArithmeticException obj){
		System.out.println("Exception occured" +obj);
	}

	finally{
		System.out.println("Terminated application");
	}
}
}