//Input: 4 4                Output:  # # # #
//                                   * * * *
//                                   # # # #
//                                   * * * *
#include<stdio.h>
void Pattern(int iRow,int iCol){
        int i,j;
        for(i=1;i<=iRow;i++){
            for(j=1;j<=iCol;j++){
                if((i%2) == 0){   //* he divisible honrrr
                    printf("*\t");
                }
                else{
                    printf("#\t");
                }
            }
            printf("\n");
        }

}
int main(){
    int iValue1=0,iValue2=0;
    printf("Enter row and column:\n");
    scanf("%d %d" , &iValue1,&iValue2);
    Pattern(iValue1,iValue2);
return 0;
}