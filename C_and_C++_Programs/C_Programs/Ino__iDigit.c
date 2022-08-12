//write a program ulta kryach aah 
//Input:453   //Output: 354
#include<stdio.h>
void Display(int);
int main(){
    int iValue = 0;
    printf("Enter number:\n");
    scanf("%d", &iValue);
    
    Display(iValue);
return 0 ;
}

void Display(int iNo){
    int iDigit ,isum=0;

    while(iNo!=0){
        iDigit = iNo%10;
        printf("%d",iDigit);//or
        //reverse = reverse*10+iDigit;
        iNo=iNo/10;
        
        
    }
    
} 
/*
//write a program ulta kryach aah  addition sum kryach
//Input:453   //Output: 354
#include<stdio.h>
int Display(int);
int main(){
    int iValue = 0 , iRet =0;
    printf("Enter number:\n");
    scanf("%d", &iValue);

    iRet = Display(iValue);
    printf("Addition of digits %d\n", iRet);
    
return 0 ;
}

int Display(int iNo){
    int iDigit = 0 ,iSum =0;

    while(iNo!=0){
        iDigit = iNo % 10;
        iSum = iSum + iDigit;
        iNo = iNo/10;
    }
 return iSum;  
} */