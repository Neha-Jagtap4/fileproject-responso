#include<iostream>

using namespace std;
 int srtlen(char *str){
     int icnt=0;
     while(*str != '\0'){
         icnt++;
         str++;
     }
     return icnt;
 }

 int sretx(char *str){
     static int icnt =0;
     if(*str != '\0'){
         icnt++;
         str++;
         sretx(str);
     }
     return icnt;
 }

 int CapCount(char *str){
     static int icnt=0;
     if(*str != '\0'){
        if((*str >='A') && (*str <= 'Z')){
            icnt++;
        }
        str++;
        CapCount(str);
        }
        return icnt;
 }
int main(){
    char Arr[20];
    cout<<"Enter string : ";
    scanf("%[^'\n']s",Arr);

    int ret= CapCount(Arr);
    cout<<"COUNT OF STRING IS :"<<ret<<"\n";

    return 0;
}