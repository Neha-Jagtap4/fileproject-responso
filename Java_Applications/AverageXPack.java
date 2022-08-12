import java.util.*;
import MarvellousNumbers.AverageX;

class Marvellous extends AverageX{
	public Marvellous(int iValue){
		super(iValue);
	}
	public float Avg(){
		int iSum=0,iCnt=0;
		for(iCnt=0;iCnt<Arr.length; iCnt++){
			iSum=iSum+Arr[iCnt];
		}
		return (iSum/Arr.length);
	}	
}

class AverageXPack{
	public static void main(String arg[]){
	Scanner sobj =new Scanner(System.in);
	int iLength=0;
	float fRet=0.0f;

	System.out.println("Enter the 0 elements: ");
	iLength = sobj.nextInt();

	Marvellous mobj =new Marvellous(iLength);
	mobj.Accept();
	mobj.Display();
	fRet=mobj.Avg();

	System.out.println("AVaerge of number is "+fRet);
	}
}