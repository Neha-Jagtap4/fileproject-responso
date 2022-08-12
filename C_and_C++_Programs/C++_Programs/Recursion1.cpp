#include<iostream>

using namespace std;

void DisplayI(int iNo){
    int iCnt=1;//auto storge classvariables number assta

    while(iCnt<=iNo){
        iCnt++;
        cout<<"Marvellous\n";
        
    }
}

void DisplayR(int iNo){
    static int iCnt=1;  //preserve number assta
    if(iCnt<=iNo){
        iCnt++;
        cout<<"Marvellous\n";
        DisplayR(iNo); //recursive cha call
    }
}

int main(){
    int i=5;
//    DisplayI(i);
    DisplayR(i);
    return 0;
}