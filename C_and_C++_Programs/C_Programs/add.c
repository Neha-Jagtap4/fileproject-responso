#include<stdio.h>
void display(int iValue);  //declaration

int main(){
    int iNo = 0;
    display(iNo);  //function call

return 0;
}

void display(int iValue){  //function definition
    int i =0;

    for (int i = 5; i<=iValue; i++) 
    {
       printf("%d\n",i,i+1);
       i++;
    }
   
    
}