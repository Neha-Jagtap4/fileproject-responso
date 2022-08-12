//Input: 3 (row)  3 (column)          o/p : 1 2 3 
//                                          1 2 3 
//      as a j                              1 2 3

#include<stdio.h>

void Pattern(int iRow,int iCol){
    int rw=0,col=0;
    for(rw=1;rw<=iRow;rw++){
        for(col=1;col<=iCol;col++){
            printf("%d\t",col);
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