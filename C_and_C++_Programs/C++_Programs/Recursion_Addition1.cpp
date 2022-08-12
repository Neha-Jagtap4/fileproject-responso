#include<iostream>
using namespace std;

int AddX(int Arr[],int iSize){
    int iSum=0, i=0;
    if(i<iSize){
        iSum = iSum + Arr[i];
        i++;
        AddX(Arr,iSize);
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
return 0;
}