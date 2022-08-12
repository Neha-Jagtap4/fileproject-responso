#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
#include<fcntl.h>
//#define O_RDWR
int main(){
    int fd=0;
    char Arr[10];

    fd=open("Neha.txt",O_RDWR);
    if(fd==-1){
        printf("Unable to open file\n");
    }
    
    lseek(fd,5,0);  //kitine pudhe jayach aah  //offset sarkhh

    read(fd,Arr,5);  //Ani 5 cha pudheche read kr op madhe
    write(1,Arr,5);
    printf(" \n Data from file is :\n");

    lseek(fd,10,0);
    read(fd,Arr,5);
    write(1,Arr,5);
    printf("\nData from file is :\n");

    close(fd);

   
return 0;
}