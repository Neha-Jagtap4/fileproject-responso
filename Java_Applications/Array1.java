import java.util.*;

class Array1{
	public static void main(String[] args){
	int arr[] = new int[5];

	arr[0]=11;
	arr[1]=21;
	arr[2]=51;
	arr[3]=101;
	arr[4]=111;

	System.out.println("Length of array is :" +arr.length);

	int brr[]={11,21,51,101,111};
	int arrr[] = new int[]{11,21,51,101,111};

	//int iCnt=0;
	for(int iCnt=0;iCnt<arrr.length;iCnt++){
		System.out.println(arrr[iCnt]);
	}

	int i=0;
	while(i<arrr.length){
		System.out.println(arrr[i]);
		i++;
	}

	int j=0;
	do{
		System.out.println(arrr[j]);
		j++;
	}while(j<arrr.length);
}
}