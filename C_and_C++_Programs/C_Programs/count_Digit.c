//Count the digit which are greater than 5.

#include<stdio.h>

int CountDigit(int iNo){
    int iDigit=0,iCount=0;
    while(iNo!=0){
        iDigit = iNo%10;
        if(iDigit>=5)
        iCount++;
        iNo=iNo/10;
    }
return iCount;
}
int main(){
    int iValue=0;
    int iRet=0;
    printf("Enter number:\n");
    scanf("%d", &iValue);
    iRet= CountDigit(iValue);
    printf("The total digit gretater than 5 is %d",iRet);

return 0;
}