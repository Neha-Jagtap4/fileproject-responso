//oVERIDDING HOTTYMEANIS  Marvellous.....Hellollous
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
#include<fcntl.h>
//#define O_RDWR
int main(){
    int fd=0;
    char Arr[]="Hello";

    fd=open("Neha.txt",O_RDWR);
    if(fd==-1){
        printf("Unable to open file\n");
    }

    write(fd,Arr,5);
    close(fd);

   
return 0;
}