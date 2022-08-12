import java.util.*;
class Add1{

	public static int Addition(int iNo1,int iNo2){
		int iAns=0;
		iAns=iNo1+iNo2;
		return iAns;
	}

	public static void main(String args[]) 
	{
		int iValue1=0,iValue2=0,iRet=0;
		Scanner sobj=new Scanner(System.in);

		System.out.println("Enter the first num");
	    iValue1=sobj.nextInt();

		System.out.println("Enter the second num");
		iValue2=sobj.nextInt();

		iRet=Addition(iValue1,iValue2);
		System.out.println("Addition is : " +iRet);

	}
}