//Write a program of printing pattern
//I/P:5        O/P: *  *  *  *  * 

#include<stdio.h>

void Display(int iNo){
    int iCnt =0;
    for(iCnt=1;iCnt<=iNo;iCnt++){
        printf("*\t");
    }
}
int main(){
    int iValue=0;
    printf("Enter the number:\n");
    scanf("%d" , &iValue);
    Display(iValue);

return 0;
}

