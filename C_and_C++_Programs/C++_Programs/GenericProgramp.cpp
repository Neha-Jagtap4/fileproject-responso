
#include<iostream>

using namespace std;

//Specific Function
void Swap(char *p,char *q){
    char temp;

    temp = *p;
    *p = *q;
    *q = temp;
}

//Generic Function
template<class T>
void SwapX(T *p,T *q){
    T temp;

    temp = *p;
    *p = *q;
    *q = temp;
}

int main(){
    char x=0, y=0;

    cout<<"Enter first number";
    cin>>x;

    cout<<"Enter Second Number\n";
    cin>>y;

    //Swap(&x,&y);
    SwapX(&x,&y);

    cout<<"Value of X becomes :"<<x<<"\n";
    cout<<"Value of Y becomes :"<<y<<"\n";

    int i=0, j=0;
    cout<<"Enter first number";
    cin>>i;

    cout<<"Enter Second Number\n";
    cin>>j;

    //Swap(&x,&y);
    SwapX(&i,&j);

    cout<<"Value of X becomes :"<<i<<"\n";
    cout<<"Value of Y becomes :"<<j<<"\n";
    return 0;



}