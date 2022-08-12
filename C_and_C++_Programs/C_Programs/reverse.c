//Reverse 
//Input : 152       //Output:251

#include<stdio.h>

int DisplayRev(int iNo){
    int iDigit = 0, iReverse = 0;
    if(iNo!=0){
        iNo = - iNo;
    }
    while(iNo!=0){
        iDigit = iNo%10;
        iReverse = iReverse*10+iDigit;
        iNo = iNo/10;
    }
 return iReverse;
}
int main(){
    int iValue = 0,iRet =0;
    printf("Enter the number\n");
    scanf("%d" ,&iValue);
    iRet = DisplayRev(iValue);
    printf("The reverse number is %d" , iRet);

return 0;
}
