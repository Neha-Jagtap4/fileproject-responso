import java.util.*;

class Add{
	public static void main(String arg[]){
		int Arr[];
		Scanner sobj =new Scanner(System.in);
		int iSize=0,i=0,iSum=0;

		Arr=new int[iSize];

		System.out.println("Enter number: ");
		iSize=sobj.nextInt();

		System.out.println("Enter the elements: ");
		for(i=0;i<Arr.length;i++){
			Arr[i]= sobj.nextInt();
		}

		for(i=0;i<Arr.length;i++){
			iSum=iSum+Arr[i];
		}

		System.out.println("Addition is : "+iSum);
	}
}