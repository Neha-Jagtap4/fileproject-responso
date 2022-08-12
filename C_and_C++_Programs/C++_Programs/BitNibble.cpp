#include<iostream>
using namespace std;
typedef unsigned int UINT;
UINT ToggleNibble(UINT iNo){
    UINT iMask1=0x01000010;
    //if we wanr to change the nibble the enter number:5,   0x50000005; 
    UINT iResult=0;

    iResult=iNo^iMask1;
    return iResult;

}

int main(){
    UINT iValue=0,iRet=0;

    cout<<"Enter Number\n";
    cin>>iValue;

    iRet=ToggleNibble(iValue);

    cout<<"Updated number is :"<<iRet<<"\n";


    return 0;
}