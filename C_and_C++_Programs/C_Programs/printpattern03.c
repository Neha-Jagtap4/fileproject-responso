//Input: 6                  Output: * # * # *
//iCnt:1 2 3 4 5 6
#include<stdio.h>

void Display(int iNo){
    int iCnt =0;
    for(iCnt=1;iCnt<=iNo;iCnt++){
        if((iCnt%2) == 0)
        printf("#\t");
        else{
            printf("*\t");
        }
        
    }
}
int main(){
    int iValue=0;
    printf("Enter number:\n");
    scanf("%d",&iValue);
    Display(iValue);

return 0;
}