#include<stdio.h>
float Average(int iNo1,int iNo2,int iNo3){
    int iSum=0;
    float res=0.0;
    iSum=iNo1+iNo2+iNo3;
    res=iSum/3;
return res;
}
int main(){
    int iValue1=0,iValue2=0,iValue3=0;
    float fRet=0.0;
    printf("Enter First numbeR:\n");
    scanf("%d",&iValue1);
    printf("Enter Second numbeR:\n");
    scanf("%d",&iValue2);
    printf("Enter Third numbeR:\n");
    scanf("%d",&iValue3);
    fRet = Average(iValue1,iValue2,iValue3);
    printf("The avgerage of sum is %f\n" ,fRet);
    

return 0;
}