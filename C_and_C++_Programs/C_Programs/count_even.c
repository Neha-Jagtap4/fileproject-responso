#include<stdio.h>
//int CountEven();
//int CountEven(){
int CountEven(int iNo){
        int count=0,iDigit =0,iSum =0 ;
        /*if(iNo == 0){
            return 1;
        }*/
        if(iNo<0){
            iNo = -iNo;
        }
        while(iNo!=0){
            iDigit = iNo%10;
            if((iDigit%2 )== 0){
            //count++;// nmuber of digit count
            iSum = iSum + iDigit;
            
            }
            iNo = iNo/10;
            //iSum = iSum + iDigit;
        }

return iSum;
}
int main(){
    int iValue = 0,  iRet = 0;
    printf("Enter number\n");
    scanf("%d" , &iValue);
    iRet = CountEven(iValue);
   // printf("Number of even digits are %d\n",iRet);
    printf("The addition of even digit is %d\n",iRet);
return 0;
}