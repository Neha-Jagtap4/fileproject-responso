#include<iostream>

using namespace std;

void Swap(char *p,char *q){
    char temp;

    temp = *p;
    *p = *q;
    *q = temp;
}

int main(){
    char x=0, y=0;

    cout<<"Enter first number";
    cin>>x;

    cout<<"Enter Second Number\n";
    cin>>y;

    Swap(&x,&y);

    cout<<"Value of X becomes :"<<x<<"\n";
    cout<<"Value of Y becomes :"<<y<<"\n";

    return 0;



}