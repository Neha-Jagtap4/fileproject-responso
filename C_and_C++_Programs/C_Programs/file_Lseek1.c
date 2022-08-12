//abcdefghijklmnopqrstuvwyz  
//op :uvwyz
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
    
    printf("Data from file is :\n");

    lseek(fd,-5,2);  //kitine maghe jayach  "uvwxyz" //offset sarkhh

    read(fd,Arr,5);  //Ani 5 cha last cha read kr op madhe

    write(1,Arr,5);

    close(fd);

   
return 0;
}