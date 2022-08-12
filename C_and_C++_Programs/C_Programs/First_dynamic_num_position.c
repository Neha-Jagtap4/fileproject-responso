//Write a program to find first Occurnce at index.
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

int FirstOcc(int Arr[],int iLength,int iNo){
    int iCnt=0;
    for(iCnt=0;iCnt<iLength;iCnt++){
        if(Arr[iCnt] == iNo){
            break;
        }
    }
    if(iCnt == iLength)
    {
        return -1;
    }
    else{
        return iCnt;
    }
}

int main(){
    int iSize=0,iCnt=0,iValue=0,iRet=0;
    int *p = NULL;

    printf("Enter the number of elemnts:\n");
    scanf("%d",&iSize);

    p = (int*)malloc(iSize*sizeof(int));

    printf("Enter elements:\n");
    for(iCnt=0;iCnt<iSize;iCnt++){
        printf("Enter the elemnts: %d", iCnt+1);
        scanf("%d" ,&p[iCnt]);
    }

    printf("Enter the element to search:\n");
    scanf("%d", &iValue);

    iRet= FirstOcc(p,iSize,iValue);

    if(iRet == -1){
        printf("Number is not there\n");
    }
    else{
        printf("Number is  there  : %d\n",iRet);
    }

    free(p);

return 0;
}