//Input: 5               Output: 2  4  6  8  10

#include<stdio.h>

void Display(int iNo){
    int iCnt =0;
    for(iCnt=1;iCnt<=iNo;iCnt++){
        iCnt = iCnt*2;
        printf("%d\t" ,iCnt);
    }
}
int main(){
    int iValue=0;
    printf("Enter number:\n");
    scanf("%d",&iValue);
    Display(iValue);

return 0;
}