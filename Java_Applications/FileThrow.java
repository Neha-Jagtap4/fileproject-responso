import java.util.Scanner;

class AgeInvalid extends Exception{
	public AgeInvalid(String str){
		super(str);
	}
}

class FileThrow{
	public static void main(String arg[]){
		Scanner sobj = new Scanner(System.in);
		System.out.println("Enter age");
		int value = sobj.nextInt();

	try{
		if(value<18){
			throw new AgeInvalid("Age us less than 18");
		}
		else{
			System.out.println("Agle is valid");
		}
	}
	catch(AgeInvalid obj){
		System.out.println("Exception occured" +obj);

	}
	finally{
		System.out.println("Terminated application");
	}
}
	
}
