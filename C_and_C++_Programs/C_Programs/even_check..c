#include<stdio.h>  //printf anf scanf
#include<stdbool.h>
bool Checkeven(int);

int main(){
    int iNo = 0;
    bool bRet = false;

    printf("Enter the number\n");
    scanf("%d", &iNo);
    bRet = Checkeven(iNo);
    bool Checkeven(int iNo);
    {
        if(bRet==true){
            printf("Number is even",iNo);
        }
        else{
            printf("Number is not even",iNo);
        }
    }
return 0;
}

bool Checkeven(int iValue){
    if((iValue%2)==0){
        return true;
    }
    else{
        return false;
    }
}