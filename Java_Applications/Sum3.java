import java.util.*;

class Sum3{
	public static int Addition(int iNo1,int iNo2){
		int iAns=0;
		iAns = iNo1+iNo2;
		return iAns;

	}
	public static void main(String arg[]){
		Scanner sobj = new Scanner(System.in);

		System.out.println("Enter first number");
		int iValue1 = sobj.nextInt();

		System.out.println("Enter second number");
		int iValue2 = sobj.nextInt();

		int iRet = Addition(iValue1,iValue2);
		System.out.println("Addition is: " +iRet);
	}
}