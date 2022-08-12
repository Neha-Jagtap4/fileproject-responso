//Count the number of digits 
//i/p : 45 o/p :2
#include<stdio.h>
int Counterdigit(int);
int main(){
    int iValue = 0 , iRet =0;
    printf("Enter number:\n");
    scanf("%d", &iValue);

    iRet = Counterdigit(iValue);
    printf("Number of digits %d\n", iRet);
    //printf("Count the number %d");
return 0 ;
}

int Counterdigit(int iNo){
    int iDigit = 0 ,iCnt = 0;
    if(iNo == 0){
        return 1;
    }    
    if(iNo<=0){
        iNo = -iNo;
    }
    while(iNo!=0){
        iDigit = iNo % 10;
        iCnt++;
        iNo = iNo/10;
        //iNo++;
    }
 return iCnt;  
} 