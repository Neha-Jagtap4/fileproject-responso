import java.util.*;

class CommandLine{
	public static void main(String arg[]){
		Scanner sobj = new Scanner(System.in);
		System.out.println("Enter total number of argumnets:" +arg.length);
		System.out.println("Command line argumnets are :");
	
	for(int i=0; i<arg.length; i++){
		System.out.println(arg[i]);
	}
}
	
}