//Input: 5 row, col Output:         * 
                //                * * *    
                //              * * * * *
                //            * * * * * * * 
                //          * * * * * * * * *        

//Complexity minimum kelli  aah

#include<stdio.h>

void Patttern(int iRow,int iCol){
    int i=0,j=0 , k=0;
    for(i=1;i<=iRow;i++){
        for(j=1;j<=iRow-i;j++){
            printf(" ");
        }
        for(k = 1; k<=2*i-1;k++){
            printf("*");
        }
        printf("\n");
    }
}
int main(){
    int iValue1=0,iValue2=0;
    printf("Enter Number:\n");
    scanf("%d %d" , &iValue1,&iValue2);
    Patttern(iValue1,iValue2);
return 0;
}