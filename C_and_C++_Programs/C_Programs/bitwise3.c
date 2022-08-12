//Check whether 7th and 12th bit is on or off.
#include<stdio.h>
#include<stdbool.h>
typedef unsigned int UINT;
bool CheckBit(int iNo){
    UINT iMask=0x000000840; //7th and 12th
    //UINT iMask=0x00000004;  //bit 4th and 15 enter number
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
    UINT iValue=0;
    bool bRet=false;

    printf("Enter number:\n");
    scanf("%d",&iValue);
    bRet=CheckBit(iValue);

    if(bRet == true){
        printf("7th and 12th  bit is ON\n");
    }
    else{
        printf("7th and 12th bit is OFF\n");
    }
return 0;
}