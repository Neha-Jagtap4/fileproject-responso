#include<iostream>

using namespace std;

void DisplayI(){
    int iCnt=1;//auto storge classvariables number assta

    while(iCnt<=4){
        iCnt++;
        cout<<"Marvellous\n";
        
    }
}

void DisplayR(){
    static int iCnt=1;  //preserve number assta
    if(iCnt<=4){
        iCnt++;
        cout<<"Marvellous\n";
        DisplayR(); //recursive cha call
    }
}

int main(){

    DisplayR();

    return 0;
}