#include<iostream>
using namespace std;
int NumAdd(int Arr[], int size){
    int isum=0;
    for(int i=0; i<size; i++){
        isum = isum + Arr[i];
    }
    return isum;
}

int NumAddX(int Arr[] ,int size){
    static int isum =0,i=0;
    
    if(i<size){
        isum = isum + Arr[i];
        i++;
        NumAddX(Arr,size);
    }
    return isum;
}
int main(){
    int iLength=0;
    cout<<"Enter number\n";
    cin>>iLength;

    int *ptr = NULL;
    ptr = new int[iLength];
    cout<<"enter elments :";
    for(int i=0; i<iLength; i++){
        cin>>ptr[i];
    }

    int iret = NumAddX(ptr,iLength); 
    cout<<"Additio is: "<<iret<<"\n";
}
