//Input : 3
//Output : 6
//Input : 5//Output :120
#include<stdio.h>

typedef unsigned long int UINT;

UINT Factorial(int);
int main(){
    int iNo= 0;
    UINT iRet = 0;

    printf("Enter the number\n");
    scanf("%d", &iNo);

    iRet = Factorial(iNo);
    printf("Factorial is : %d", iRet);
return 0;
}

UINT Factorial(int iValue){
    UINT iFact =1;
    int iCnt = 0;

    if(iValue<0){
        iValue = - iValue;
    }

    iCnt = 1;
    while(iCnt<=iValue){
        iFact = iFact*iCnt;
        iCnt++;
    }
    return iFact;
}

