import java.util.*;

class ArrayX{
	public int Arr[];

	public ArrayX(int iSize){  //Characteritics 
		Arr = new int[iSize];
	}

	public void Accept(){     //Behaviour
		Scanner sobj=new Scanner(System.in);
		int iCnt=0;
		System.out.println("Enter elements:");
		for(iCnt=0; iCnt <Arr.length; iCnt++){
			Arr[iCnt]=sobj.nextInt();
		}	
	}

	public void Display(){   //Behaviour
		int iCnt=0;
		System.out.println("Elements are: ");
		for(iCnt=0;iCnt<Arr.length;iCnt++){
			System.out.println(Arr[iCnt]);
		}		
	}
}

class Marvellous extends ArrayX{
	public Marvellous(int iValue){
		super(iValue);
	}
	public int Add(){
		int iSum=0,iCnt=0;
		for(iCnt=0;iCnt<Arr.length; iCnt++){
			iSum=iSum+Arr[iCnt];
		}
		return iSum;
	}	
}

class AdditionArray1{
	public static void main(String arg[]){
	Scanner sobj =new Scanner(System.in);
	int iLength=0,iRet=0;

	System.out.println("Enter the 0 elements: ");
	iLength = sobj.nextInt();

	Marvellous mobj =new Marvellous(iLength);
	mobj.Accept();
	mobj.Display();
	iRet=mobj.Add();

	System.out.println("Addition is "+iRet);
	}
}