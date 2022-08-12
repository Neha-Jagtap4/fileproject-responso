//Write a dynamic user provided number int Arr[],int iLength,int iNo
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

bool CheckNumber(int Arr[],int iLength,int iNo){
    int iCnt=0;
    for(iCnt=0;iCnt<iLength;iCnt++){
        if(Arr[iCnt] == iNo){
            break;
        }
    }
    if(iCnt == iLength)
    {
        return false;
    }
    else{
        return true;
    }
}

int main(){
    int iSize=0,iCnt=0,iValue=0;
    int *p = NULL;
    bool bRet = false;

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

    bRet= CheckNumber(p,iSize,iValue);

    if(bRet == true){
        printf("Number is there\n");
    }
    else{
        printf("Number is not there\n");
    }

    free(p);

return 0;
}