#include<stdio.h>

void display();

int main(){
    int iNo = 0;
    printf("Enter number :");
    scanf("%d",&iNo);

    display(iNo);
return 0;
}


void display(int iValue){
    int i = 0;
    if(iValue ==0){
        printf("your entered number is 0");
        return;
    }
    if(iValue<0){      //input updater negative convert to positive
        iValue = -iValue;
    }
    for(int i = 0;i<=iValue;i++){  //0............5 print hoil
        printf("%d",i);
    }

}