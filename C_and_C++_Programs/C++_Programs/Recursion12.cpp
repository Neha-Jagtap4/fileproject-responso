#include<iostream>
using namespace std;

int Factorial(int ino){
    int iFact=1;
    while(ino != 0){
        iFact = iFact * ino;
        ino--;
    }
    return iFact;
}

int FactorialR(int no){
    static int iFact =1;
    if(no != 0){
        iFact = iFact * no;
        no--;
        FactorialR(no);
    }
    return iFact;
}
int main(){
    int x=0;
    cout<<"Enter number\n";
    cin>>x;

    int iret  = FactorialR(x);
    cout<<"Factorial is : "<<iret<<"\n";

    return 0;
}