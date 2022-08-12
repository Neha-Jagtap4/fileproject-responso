//print : * * * *
//        * * * *
//        * * * *

#include<stdio.h>

void Pattern(int iRow,int iCol){
    int rw=0,col=0;
    for(rw=1;rw<=iRow;rw++){
        for(col=1;col<=iCol;col++){
            printf("*\t");
        }
    printf("\n"); 
    }

}

int main(){
    int iValue1=0,iValue2=0;
    printf("Enter row :\n");
    scanf("%d %d" ,&iValue1, &iValue2);
    Pattern(iValue1,iValue2);

return 0;
}