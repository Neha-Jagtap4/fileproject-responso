//print : * * * *
//        * * * *
//        * * * *
       
#include<stdio.h>

void Display(){
    int iRow=3;
    int iCol=4;
    int i,j;
    for(i = 1;i<=iRow;i++){
        for(j=1;j<=iCol;j++){
            printf("*\t");
        }
        printf("\n");
    }
}
int main(){
    Display();
return 0;
}