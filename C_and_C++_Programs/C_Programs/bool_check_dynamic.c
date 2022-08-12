#include<stdio.h>
//Write a program th check whether the 11 is present or not.

#include<stdlib.h>
#include<stdbool.h>

bool CheckNumber(int Arr[],int iLength){
    int iCnt=0;
    bool bAns = false;

    for(iCnt=0;iCnt<iLength;iCnt++){
        if(Arr[iCnt] == 11){
            bAns =true;
            break;
        }
    }
    return bAns;

}

int main(){
    int iSize=0,iCnt=0;
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

    bRet= CheckNumber(p,iSize);

    if(bRet == true){
        printf("Number is there\n");
    }
    else{
        printf("Number is not there\n");
    }

    free(p);

return 0;
}