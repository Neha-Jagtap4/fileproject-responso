#include<iostream>

using namespace std;
class Marvellous{
    public:
    int Addition(int x,int y){
        int iResult=0;
        iResult=x+y;
        return iResult;
    }
};

int main(){
    int iNo1=0,iNo2=0,iRet=0;
    Marvellous mobj;

    cout<<"Enter first number\n";
    cin>>iNo1;

    cout<<"Enter second number\n";
    cin>>iNo2;

    iRet=mobj.Addition(iNo1,iNo2);

    cout<<"ADDITION IS :"<<iRet<<"\n";

return 0;
}