//Write a program which count the even number digit
#include<stdio.h>
#include<stdlib.h>

int CountEven(int Arr[], int iLength){
    int iCnt=0,iSum=0,iCount=0;
    for(iCnt=0;iCnt<iLength;iCnt++){
        if(Arr[iCnt]%2 == 0){
            //iSum=iSum+ Arr[iCnt];///////HE addition satthi count chi
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

    iRet= CountEven(p,iSize);

    printf("the Count Even nnumbers are : %d" ,iRet);

    free(p);

return 0;
}