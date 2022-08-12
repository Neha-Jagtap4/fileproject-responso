//Neha.txt file create
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
#include<fcntl.h>
//#define O_RDWR
int main(){
    int fd=0,iRet=0;
    char Fname[30];
    char Data[]="Marvellous Hellos";

    printf("Enter file name\n");
    scanf("%s",Fname);
    
    fd=open(Fname,O_RDWR);

    if(fd == -1){
        printf("Unable to OPEN the files\n");
        //return -1;
    }
    else{
        printf("File sucessfully Opened with fd :%d\n",fd);
    }

    iRet=write(fd,Data,10);
    printf("%d bytes gets suceesfully written in the file\n",iRet);
return 0;
}