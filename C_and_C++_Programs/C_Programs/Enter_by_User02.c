//Input : 3
//Output : 1 2 3

////Input : 7
//Output : 1 2 3 4 5 6 7
#include<stdio.h>

void display(int);

int main(){
    int iNo = 0 ;
    printf("Enter the number :");
    scanf("%d",&iNo);

    display(iNo);
return 0;
}

void display(int n){
    int i =0;
    for(int i = 1; i<= n;i++){
        printf("%d\n",i);
    }
}