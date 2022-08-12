/*#include<stdio.h>
int main(){
    char ch1 ='A';
    char ch2 ='a';
    char ch3 ='0';

    printf("AsSCII value valoue of ch1 is : %d\n", ch1);
    printf("AsSCII value valoue of ch2 is : %d\n", ch2);
    printf("AsSCII value valoue of ch3 is : %d\n", ch3);
return 0; //output 65  97  48
}*/

#include<stdio.h>

void DisplayTable(){
    int iCnt=0;

    printf("ASCII table is\n");
    printf("****************************\n");
    printf("Decimal\tHexdecimal\tOctal\tcharacter\n");
    printf("***************************\n");

    for(iCnt=0;iCnt<127;iCnt++){
        printf("%d\t%x\t%o\t%c\t\n",iCnt,iCnt,iCnt,iCnt);
    }
}
int main(){
        DisplayTable();
return 0;
}