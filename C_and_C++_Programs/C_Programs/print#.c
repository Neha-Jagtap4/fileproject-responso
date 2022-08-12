//print # 5 times........
#include<stdio.h>

void display();  //declaration
int main(){
    display();//function call....it does not display if don't delecare the display the complier does not understand
    return 0;
}
void display(){  //definition

    int i = 0 ;
    for(int i = 1 ; i<= 5; i++){
        printf("#\n");
    }
}