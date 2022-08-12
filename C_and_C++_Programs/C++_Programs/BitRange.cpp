//Input: 10  ,Position:4   .Output:2
#include<iostream>
using namespace std;
typedef unsigned int UINT;
UINT ToggleRange(UINT iNo,int iStart,int iEnd)
{
    UINT iMask1=0x55,iMask2=0xFFFFFFFF,iMask=0; //iMask1=0x0000000B,iMask2=0x0000000B
    UINT iResult=0;

    iMask1=iMask1<<(iStart-1);
    iMask2=iMask2>>(32-iEnd);
    iMask=iMask1&iMask2;
    iResult=iNo^iMask;
    return iResult;
    
    
}

//iStart =5
//iEnd: 16

int main(){
    int iValue=0,iRet=0,i=0,j=0;

    cout<<"Enter number\n";
    cin>>iValue;

    cout<<"Enter start position\n";
    cin>>i;

    cout<<"Enter end position\n";
    cin>>j;

    iRet=ToggleRange(iValue,i,j);

    cout<<"Updated number is :"<<iRet<<"\n";

return 0;
}