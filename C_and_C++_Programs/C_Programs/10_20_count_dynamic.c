//Write aprogram which which count the number present between 10 and 20.

#include<stdio.h>
#include<stdlib.h>

int CountRange(int Arr[], int iLength){
    int iCnt=0,iCount=0;
    for(iCnt=0;iCnt<iLength;iCnt++){
        if(Arr[iCnt] > 10 && Arr[iCnt]<20){
            
            iCount++;
        }
    }
return iCount;
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

    iRet= CountRange(p,iSize);

    printf("the Count nnumbers are : %d" ,iRet);

    free(p);

return 0;
}