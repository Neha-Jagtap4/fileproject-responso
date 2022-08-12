//Find the minimum number in program

#include<stdio.h>
#include<stdlib.h>

int Minimum(int Arr[], int iLength){
    int iCnt=0,iMini=Arr[iCnt];
    for(iCnt=0;iCnt<iLength;iCnt++){
        if(Arr[iCnt] < iMini){
            
            iMini= Arr[iCnt];
        }
    }
return iMini;
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

    iRet= Minimum(p,iSize);

    printf("the Minimum numbers are : %d" ,iRet);

    free(p);

return 0;
}