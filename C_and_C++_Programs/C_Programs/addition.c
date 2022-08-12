
#include<stdio.h>

int Addition(int ino1,int ino2){
    return ino1+ino2;
}

int main(){
    int x,y,z;
    printf("Enter first number :");
    scanf("%d", &x);

    printf("Enter second number :");
    scanf("%d", &y);

    z = Addition(x,y);

    printf("Addition %d",z);
return 0;
}

