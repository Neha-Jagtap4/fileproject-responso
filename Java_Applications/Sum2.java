import java.util.*;

class Summing{
	int i=0;
	//public int arr[] = new int (iLength); hw nahi acees honr
	public int arr[];
	public Summing(int iLength){
		arr = new int[iLength];

	}

	public void Accept(){
		Scanner sobj = new Scanner(System.in);
		System.out.println("Ener no. of elements");
		for (i=0; i<arr.length; i++ ) {
			arr[i] = sobj.nextInt();
		}
	}

	public int addition(){
		int iSum=0;
		for(int i=0; i<arr.length; i++){
			iSum=iSum+arr[i];
		}
		return iSum;
	}
}

class Sum2{
	public static void main(String args[]){
		int iLength=0,iRet=0;
		Scanner sobj = new Scanner(System.in);
		System.out.println("Enter number");
		iLength = sobj.nextInt();

		Summing mobj = new Summing(iLength);
		mobj.Accept();
		iRet = mobj.addition();
		System.out.println("Addition is : "+iRet);

	}
}