//Input: 5 row, col Output: * * * * *
                //          * * * *
                //          * * *
                //          * *
                //          *

//Complexity minimum kelli  aah

#include<stdio.h>

void Patttern(int iRow,int iCol){
    int i=0,j=0;
    for(i=iRow;i>=1;i--){
        for(j=i;j>=1;j--){
                printf("*");
        }
    printf("\n");
    }
}
int main(){
    int iValue1=0,iValue2=0;
    printf("Enter Number:\n");
    scanf("%d" , &iValue1,&iValue2);
    Patttern(iValue1,iValue2);
return 0;
}