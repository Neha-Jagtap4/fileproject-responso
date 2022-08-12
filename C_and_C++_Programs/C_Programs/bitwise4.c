#include<stdio.h>
#include<stdbool.h>
typedef unsigned int UINT;

bool CheckBit(int iNo){
    UINT iMask=0x00000001;
    UINT iResult=0;

    iResult =iNo&iMask;

    if(iResult==iMask){
        return true;
    }
    else{
        return false;
    }
}

int main(){
    UINT iValue=0, iBit=0;
    bool bRet=false;

    printf("Enter number:\n");
    scanf("%d",&iValue);
    printf("Enter bit:\n");
    scanf("%d",&iBit);
    bRet=CheckBit(iValue);

    if(bRet == true){
        printf(" bit is ON\n");
    }
    else{
        printf(" bit is OFF\n");
    }
return 0;
}