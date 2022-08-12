#include<iostream>
#include<stdio.h>

using namespace std;

int Strelen(char *str){
    static int iCnt=0;
    if(*str != '\0'){
        iCnt++;
        str++;
        Strelen(str);
    }
    return iCnt;
}

int main(){
    char Arr[20];
    int iRet=0;

    cout<<"Enter string\n";
    scanf("%[^'\n']s",Arr);

    iRet = Strelen(Arr);
    cout<<"Length of string:"<<iRet<<"\n";

    return 0;

}