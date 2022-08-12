//Neha.txt file create
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
#include<fcntl.h>
//#define O_RDWR
int main(){
    int fd=0;
    char Fname[30];

    printf("Enter file name\n");
    scanf("%s",Fname);
    
    fd=open(Fname,O_RDWR);

    if(fd == -1){
        printf("Unable to OPEN the files\n");
    }
    else{
        printf("File sucessfully Opened with fd : %d\n",fd);
    }
return 0;
}