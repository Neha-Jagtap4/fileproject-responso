//Write a program to find maximum number.
#include<stdio.h>
#include<stdlib.h>

int Maximum(int Arr[], int iLength){
    int iCnt=0,iMax=0;
    for(iCnt=0;iCnt<iLength;iCnt++){
        if(Arr[iCnt] > iMax){
            
            iMax= Arr[iCnt];
        }
    }
return iMax;
}
int main(){
    int iSize=0,iCnt=0,iRet=0;
    int *p = NULL;

    printf("Enter the number of elemnts:\n");
    scanf("%d",&iSize);

    p = (int*)malloc(iSize*sizeof(int));

    printf("Enter elements:\n");
    for(iCnt=0;iCnt<iSize;iCnt++){
        printf("Enter the elemnts: %d", iCnt+1);
        scanf("%d" ,&p[iCnt]);
    }

    iRet= Maximum(p,iSize);

    printf("the maximum numbers are : %d" ,iRet);

    free(p);

return 0;
}