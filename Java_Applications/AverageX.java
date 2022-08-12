package MarvellousNumbers;

import java.util.*;

public class AverageX{
	public int Arr[];

	public AverageX(int iSize){  //Characteritics 
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