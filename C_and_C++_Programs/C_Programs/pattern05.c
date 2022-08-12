//Input: 5 row, col Output: 
                //            1 2 3 4
                //              2 3 4
                //                3 4
                //                  4

//Complexity minimum kelli  aah

#include<stdio.h>

void Patttern(int iRow,int iCol){
    int i=0,j=0 , k=0;
    for(i=1;i<=iRow;i++){
        for(j=1;j<=i;j++){
            printf(" ");
        }
        for(k = i; k<=iRow;k++){
            printf("%d", k);
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