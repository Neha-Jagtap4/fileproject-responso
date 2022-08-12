//Example aah 
#include<iostream>

using namespace std;

void Swap(int *p,int *q){
    int temp;

    temp = *p;
    *p = *q;
    *q = temp;
}

int main(){
    int x=0, y=0;

    cout<<"Enter first number";
    cin>>x;

    cout<<"Enter Second Number\n";
    cin>>y;

    Swap(&x,&y);

    cout<<"Value of X becomes :"<<x<<"\n";
    cout<<"Value of Y becomes :"<<y<<"\n";

    return 0;



}