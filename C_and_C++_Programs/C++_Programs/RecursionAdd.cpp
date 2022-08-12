#include<iostream>
using namespace std;

/*int Sum(int iNo){
    int iSum=0;
    while(iNo!= 0){
        iSum = iSum + (iNo%10);
        iNo = iNo/10;
    }
    return iSum;

}*/

int Sum(int iNo){
    static int iSum=0;
    if(iNo != 0){
        iSum = iSum + (iNo%10);
        iNo = iNo/10;
        Sum(iNo);
    }
    return iSum;
}

//mhanje stack frame 231 : la 3 daa zal aaha
int main(){
    int x=0, iRet=0;

    cout<<"Enter number\n";
    cin>>x;

    iRet = Sum(x);

    cout<<"Summation of didgit: "<<iRet<<"\n";


    return 0;
}

