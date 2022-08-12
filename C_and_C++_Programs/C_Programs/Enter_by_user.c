//Accept number from user and display number World on screen
#include<stdio.h>

void Display(int);

int main(){
    int iNo = 0;
    printf("Enter the number\n");
    scanf("%d",&iNo);

    Display(iNo);
return 0;
}

void Display(int iValue){
    int i = 0;
    for(int i = 0; i<=iValue; i++){
        printf("World\n");
    }
}