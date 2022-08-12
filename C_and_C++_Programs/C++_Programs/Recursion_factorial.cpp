#include<iostream>

using namespace std;

int Factorial(int iNo){
    static int iFact=1;

    if(iNo != 0){
        iFact = iFact*iNo;
        iNo--;
        Factorial(iNo);
    }
    return iFact;
}
int main(){
    int x=0, iRet=0;

    cout<<"Enter number\n";
    cin>>x;

    iRet = Factorial(x);

    cout<<"Factorial is : "<<iRet<<"\n";


    return 0;
}