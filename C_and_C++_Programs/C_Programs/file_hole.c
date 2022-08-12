//abcdefghijklmnopqrstuvwyz
//abcdefghijklmnopqrstuvwyz          *
             //space antr * yeil  10 bytes
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
    
    lseek(fd,10,2);  //kitine maghe jayach  "uvwxyz" //offset sarkhh

    //read(fd,"*",1);  //Ani 5 cha last cha read kr op madhe

    write(fd,"*",1);

    close(fd);

   
return 0;
}