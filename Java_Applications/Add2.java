import java.util.*;
class Add{

	public int arr[];
	public Add(int iSize){
		 arr = new int[iSize];
	}
	public void Accept(){
		Scanner sobj=new Scanner(System.in);
		System.out.println("Enter elements ");
		for(int i=0;i<arr.length;i++){
			arr[i]=sobj.nextInt();
		}
	}

	public int Addition(){
		int iSum=0;
		for(int i=0;i<arr.length;i++){
			iSum=iSum+arr[i];
		}
		return iSum;
	}
}

class Add2{
	public static void main(String[] args){
		Scanner sobj =new Scanner(System.in);
		System.out.println("Enter number:");
		int iSize=sobj.nextInt();

		Add obj =new Add(iSize);
		obj.Accept();

		int iRet =obj.Addition();
		System.out.println("Addition is : "+iRet);
	}
}