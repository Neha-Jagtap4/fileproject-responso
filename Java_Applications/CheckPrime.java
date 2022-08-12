import java.util.*;

class Demo{
	public boolean CheckPrime(int iNo){
		int iCnt =0;
		boolean bflag = true;

		for(iCnt=2;iCnt<=iNo/2;iCnt++){
			if((iNo%iCnt)==0){
			bflag = false;
			break;
			}
		}
		return bflag;
	}
}

class CheckPrime{
	public static void main(String arg[]){
		Scanner sobj = new Scanner(System.in);
		int iValue=0;
		boolean bRet= false;

		System.out.println("Enter the value:");
		iValue=sobj.nextInt();

		Demo dobj=new Demo();
		bRet=dobj.CheckPrime(iValue);
		if(bRet == true){
			System.out.println("Number is Prime");
		}
		else{
			System.out.println("Number is not Prime");
		}

	}
}