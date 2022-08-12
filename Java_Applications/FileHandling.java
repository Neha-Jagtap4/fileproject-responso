import java.util.*;

class FileHandling{
	public static void main(String arg[]){
		int arr[] = new int[5];  //ptr =(int *size)malloc(sizeof(int));
		arr[0]=10;arr[1]=20;
		arr[2]=30;arr[3]=40;
		arr[4]=50;

		Scanner sobj = new Scanner(System.in);
		System.out.println("Enter the Index of array");
		int index= sobj.nextInt();

		try{
			System.out.println("Data is :" +arr[index]);
		} 

		catch(ArrayIndexOutOfBoundsException obj){
			System.out.println("Exception is occured: "+obj);

		}

		finally{
			System.out.println("Application is terminated");
		}
		System.out.println("End of main");
		
	}
}