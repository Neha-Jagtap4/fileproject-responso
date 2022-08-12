0///Input :3
//Output : 6

///Input :5
//Output : 120

#include<stdio.h>
int Factorial(int);
int main(){
    int iNo= 0, iRet = 0;
    printf("Enter number :");
    scanf("%d",&iNo);
    iRet = Factorial(iNo);

    printf("Factorial is : %d", iRet);
return 0;
}                                           // _______
//isum = 0 and icnt = Start.......1 2 3 4 5    | 0 1 |  mg icnt pahile 1 add hott 0 = 1 then 1 + icnt 2 = 3....upto....15
//1 3 6 10 15   
int Factorial(int iValue){
    int iFact = 1;
    int iCnt = 0;

    if(iValue<0){
        iValue = - iValue;   //updator
    }
    for(iCnt = 1; iCnt<=iValue;iCnt++){
        iFact = iFact*iCnt;
    }

return iFact ;
}

//here we also use updator if enter value is negative then it convert into positive...
/*if(iValue<0){
    iValue = - iValue;   //updator
}*/