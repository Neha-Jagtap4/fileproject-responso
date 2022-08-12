import java.util.*;

class Sum1{
	public static void main (String arg[]){
		int inum=0, i=0,iSum=0;
		Scanner sobj = new Scanner(System.in);
		System.out.println("Enter number");
		inum = sobj.nextInt();

		int arr[] = new int [inum];//Memmoery allocation krtoy aah
		

		System.out.println("Enter the elements:");
		for(i=0; i<arr.length; i++){
			arr[i] = sobj.nextInt();
		}

		for(i=0; i<arr.length; i++){
			iSum = iSum + arr[i];
		}

		System.out.println("Addition is: " +iSum);

	}
}