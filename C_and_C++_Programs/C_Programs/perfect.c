//Number is Perfect number or not
#include<stdio.h>
#include<stdbool.h>
int SumFactors(int);
bool CheckPerfect(int);
int main(){
    int iValue = 0;
    bool bRet = false;
    printf("Enter the number:\n");
    scanf("%d" , &iValue);
    bRet = CheckPerfect(iValue);
    if(bRet == true){
        printf("Nmuner is perfect\n");
    }
    else{
        printf("Number is not Perfect\n");
    }
return 0;
}

bool CheckPerfect(int iNumber){
    int iRet =0;
    if(iNumber < 0){
        iNumber = -iNumber;
    }
    iRet = SumFactors(iNumber);
   /* if(iNumber<0){
        iNumber = -iNumber;
    }*/
    if (iRet ==iNumber){
        return true;
    }
    else{
        return false;
    }
}
int SumFactors(int iNo){
    int iSum =0 , iCnt =0;
    if(iNo<0){
        iNo = -iNo;
    }
    for(iCnt = 0;iCnt<=(iNo/2);iCnt++){
        if(iNo%2 == 0){
            iSum = iSum + iCnt;
        }
    }
    return iSum;
}
//-28error