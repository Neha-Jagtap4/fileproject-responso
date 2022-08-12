//Greater than 10 the do  addition 
#include<stdio.h>
#include<stdlib.h>

int Addgreater(int Arr[], int iLength){
    int iCnt=0,iSum=0;
    for(iCnt=0;iCnt<iLength;iCnt++){
        if(Arr[iCnt] > 10){
            iSum=iSum+ Arr[iCnt];
        
        }
    }
return iSum;
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

    iRet= Addgreater(p,iSize);

    printf("the Addition Greater numbers are : %d" ,iRet);

    free(p);

return 0;
}