//reverse enter by user fo ex. 10..............1
#include<stdio.h>
void display(int);  //declaration

int main(){
    int iNo=0;

    printf("Enter the number\n");
    scanf("%d",&iNo);

    display(iNo);
return 0;
}

void display(int iValue){
    int i =0;

    for(i = iValue;i>=1; i--){
        printf("%d\n",i);
    }
}