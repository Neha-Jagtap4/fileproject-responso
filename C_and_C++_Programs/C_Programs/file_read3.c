//Neha.txt: abcdehijklmnopqrstuvwyz
//O/P :First 5 letters
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
  
    read(fd,Arr,5);
    printf("Data from file is :\n");
    write(1,Arr,5);
    close(fd);

   
return 0;
}