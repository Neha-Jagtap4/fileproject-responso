/*#include<stdio.h>

void Display(int n){
    
    int i = 0;
    for( int i = 0; i<=n ; i++){
        printf("World\n");
    
    }
}
int main(){
    int iNo = 0;
    printf("Enter the number\n");
    scanf("%d\n",&iNo);
    Display(iNo);
return 0;
}
*/
///////////////////////////////////////////////////////////////////
#include<stdio.h>
void display(int);


int main(){
    int iNo= 0;
    printf("Enter the number :");
    scanf("%d", &iNo);
    display(iNo);
return 0 ;
}

void display(int iValue){
    int i = 0;
    for(int i = 0; i<=iValue;i++){
        printf("*\n");
    }
}