//user number in reverse order....print

#include<stdio.h>

void display();
int main(){
    
    display();

return 0;
}

void display(){
    int i = 0;
    //for(int i = 1 ; i<= 5;i++) .................1 2 3 4 5 order straight order
    for(int i =5;i>=1;i--)    //for reverse order i-- is used
    {
        printf("%d\n",i);
    }
}