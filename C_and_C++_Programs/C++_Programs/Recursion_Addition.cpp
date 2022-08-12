#include<iostream>
using namespace std;

int AddX(int Arr[],int iSize){
    int iSum=0, i=0;
    for(i=0; i<iSize; i++){
    iSum = iSum + Arr[i];
    }
    return iSum;
}

int main(){
    int iLength=0,i=0,iRet=0;
    int *p = NULL;

    p = new int[iLength];
    cout<<"Emter the number\n";
    cin>>iLength;

    
    cout<<"Enter the elements";
    for(i=0; i<iLength; i++){
        cin>>p[i];
    }

    iRet = AddX(p,iLength);
    cout<<"Addition is: "<<iRet<<"\n";

}