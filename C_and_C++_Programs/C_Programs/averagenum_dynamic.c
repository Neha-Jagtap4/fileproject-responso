//Write a program which perform averge of addition by using array, dynamic allocation
#include<stdio.h>
#include<stdlib.h>
float Average(int Arr[],int iLength){
    int iSum=0,iCnt=0;
    float  fResult = 0.0f;

    for(iCnt=0;iCnt<iLength;iCnt++){
        iSum=iSum+ Arr[iCnt];
    }

    fResult = (float)iSum/iLength;
    return fResult;

}
int main(){
    int iSize=0,iCnt=0;
    int *ptr= NULL; //p is pointer
    float fRet = 0.0f;

    printf("Enter the number of elements:\n");
    scanf("%d" ,&iSize);
    //ptr = malloc(iSize*sizeof(int)); //it contains * void but we want specific value we use (int*)
    ptr = (int*)malloc(iSize*sizeof(int));

    printf("Enter elements:\n");
    for(iCnt=0;iCnt<iSize;iCnt++){
        printf("Enter the number of elements: %d" , iCnt+1);
        scanf("%d" , &ptr[iCnt]);
    }
    printf("Enter data:\n");
    for(iCnt=0;iCnt<iSize;iCnt++){
        printf("%d\t" , ptr[iCnt]);
    }

    fRet=Average(ptr,iSize);
    printf("Avarge is :%f\n",fRet);
    free(ptr);


return 0;
}