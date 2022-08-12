//Input :120
//Output : 5 4 3 2 1
//Input : 24 //Output : 4 3 2 1

//Factors kadhn mhanje toh number kiti number sobt direct divisible hottt aahh
//eneter : 8 ......factors 1 2 4
//Ith aapn factors kadtoy aah mhanun we use void ,We don't number of factorial...
#include<stdio.h>
void DisplayFactors(int);

int main(){
    int iNo = 0;

    printf("Enter the number\n");
    scanf("%d", &iNo);

    DisplayFactors(iNo);
return 0;
}

void DisplayFactors(int iValue){
    
    int iCnt = 0;

    if(iValue<0){   ///input upadter
        iValue = -iValue;
    }

    for(iCnt = 1; iCnt<iValue; iCnt++){
            if((iValue%iCnt) == 0){
                printf("%d\n", iCnt);
        }
    }
}

/*  in this directly divisible
(8%1) == 0      1
(8%2) == 0      4
(8%3) == 0 
(8%4) == 0      2
(8%5) == 0 
(8%6) == 0 
(8%7) == 0 
*/