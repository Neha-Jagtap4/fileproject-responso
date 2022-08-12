#include<iostream>
using namespace std;

int Addition(int ino){   //auto local varaible
    int isum=0,iDigit=0;

    while(ino != 0){
        iDigit = ino%10;
        isum = isum + iDigit;
        ino = ino/10;
    }
    return isum;
}

int sumX(int ino){
    static int isum=0;
    int iDigit=0;

    if(ino != 0){
        iDigit = ino%10;
        isum = isum + ino%10;
        ino = ino/10;
        sumX(ino);

    }
    return isum;
}


int main(){
    int x=0;
    cout<<"Enter number\n";
    cin>>x;

    int iret = sumX(x);
    cout<<"Addition is :"<<iret;
    return 0;
}