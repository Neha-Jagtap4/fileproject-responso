import java.util.*;

class Demo{
	public boolean Armstrong(int iNo){
		int Temp=0,iDigcnt=0,iDigit=0,iPower=1;
		int iCnt=0,iSum=0;

		if(iNo<0){
			iNo=-iNo;
		}
		Temp=iNo;

		while(Temp!=0){
			iDigcnt++;
			Temp=Temp/10;
		}
		Temp=iNo;

		while(Temp!=0){
			iDigit = Temp%10;
			for(iCnt=1;iCnt<=iDigcnt;iCnt++){
					iPower=iPower*iDigit;
			}
			iSum=iSum+iPower;
			iPower=1;
			Temp=Temp/10;
		}

		if(iSum==iNo){
			return true;
		}
		else{
			return false;
		}
	}

}

class Armstrong{
	public static void main(String arg[]){
		Scanner sobj= new Scanner(System.in);
		int iValue=0;
		boolean bRet=false;

		System.out.println("Enter the value:");
		iValue=sobj.nextInt();

		Demo dobj=new Demo();
		bRet=dobj.Armstrong(iValue);
		if(bRet == true){
			System.out.println("Number is Armstrong");
		}
		else{
			System.out.println("Number is not Armstrong");
		}

	}
}