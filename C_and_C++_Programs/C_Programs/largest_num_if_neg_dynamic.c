//If we enter the negative value then  in max it will give 0 ,then avoid this condition the subsitute  iMax= Arr[iCnt] to gives - then it will not return zero
//Enter the elemnts: 1    -3 
//Enter the elemnts: 2     -78
//Enter the elemnts: 3     -45
//Enter the elemnts: 4    -80
//Enter the elemnts: 5      -2
//the maximum numbers are : - 2
#include<stdio.h>
#include<stdlib.h>

int Maximum(int Arr[], int iLength){
    int iCnt=0,iMax=Arr[iCnt];
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