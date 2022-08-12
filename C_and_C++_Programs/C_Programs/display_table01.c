//Input : 5          Output: 5 10 15 20 25 ......50
#include<stdio.h>
void DisplayTable(int);
int main(){
    int iValue ;
   // int iRet ;
    printf("Enter number\n");
    scanf("%d", &iValue);
    //printf("Mul table is %d X %d = %d" ,iRet);
    DisplayTable(iValue);
return 0;
}

void DisplayTable(int iNo){
    int iCnt ,itable;
    if(iNo == 0){
        return ;
    }
    if(iNo<0){
        iNo=-iNo;
    }
    for(iCnt = 1;iCnt<=10;iCnt++){
        itable = iNo*iCnt;
        printf("%d\n",itable);

    }
}