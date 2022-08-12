#include<stdio.h>

int Power(int,int);
int main(){
    int iValue1 ,iValue2 ,iRet =0;
    printf("Enter first number:");
    scanf("%d" , &iValue1);
    printf("Enter second number:");
    scanf("%d" , &iValue2);
    //display(iValue1,iValue2);
    iRet = Power(iValue1,iValue2);
    printf("The power is %d" , iRet);
    
return 0;
}
//power(2,4)......x=2,y=4
int Power(int x,int y){
    int iMult= 1,iCnt =0;

    for(iCnt = 1;iCnt <=y;iCnt++){
        iMult = iMult*x;//x is a base
    }
    return iMult;
}