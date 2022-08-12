//Reverse 
//Input : 152       //Outpu:251
//Input: -453       //Output : -354
//Here we don't use upadter so here the negative value enter in the loop and it return negative value of reverse number
#include<stdio.h>

int DisplayRev(int iNo){
    int iDigit = 0, iReverse = 0;
    
    while(iNo!=0){
        iDigit = iNo%10;
        //printf("%d" , iDigit);//.......it display the spearte sinle digit on one line
        iReverse = iReverse*10+iDigit;  //he reverese number banvtoy
        iNo = iNo/10;
    
     
    }
    
 return iReverse;
}
int main(){
    int iValue = 0,iRet =0;
    printf("Enter the number\n");
    scanf("%d" ,&iValue);
    iRet = DisplayRev(iValue);
    printf("The display reverse is %d" , iRet);

return 0;
}
