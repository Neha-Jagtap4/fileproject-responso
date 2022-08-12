import java.util.*;
class Marvellous{
	public int Add(int Brr[]){
		int iSum=0,iCNT=0;
		for(iCNT=0;iCNT<Brr.length;iCNT++){
			iSum=iSum+Brr[iCNT];
		}
		return iSum;
	}
}
class AdditionArray{
	public static void main(String arg[]){
		Scanner sobj=new Scanner(System.in);
		int iSize=0,iCnt=0,iRet=0;

		System.out.println("Enter number of elemnts");
		iSize=sobj.nextInt();

		//Arr=(int*)malloc(sizeof(int)*iSize);
		int Arr[]=new int[iSize];

		System.out.println("Enter numbers:");
		for(iCnt=0;iCnt<Arr.length;iCnt++){   //ith iSize pn lihu shkto
			Arr[iCnt] = sobj.nextInt();
			//scanf("%d",&Arr[iCnt]);
		
		}

		Marvellous mobj=new Marvellous();
		iRet=mobj.Add(Arr);   //iRet=Add(Arr,iSize);

		System.out.println("Addition is: "+iRet);

	}
}