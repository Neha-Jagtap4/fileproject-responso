#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
#include<fcntl.h>
//#define O_RDWR
int main(){
    int fd=0,iRet=0;
    char Fname[30];
    char Data[6]; //Marvellous  hellos 

    printf("Enter file name\n");
    scanf("%s",Fname);
    
    fd=open(Fname,O_RDWR);

    if(fd==-1){
        printf("Unable to OPEN the files\n");
        return -1;
    }
    else{
        printf("File sucessfully Opened with fd :%d\n",fd);
    }

    iRet=read(fd,Data,6);

    printf("%d bytes gets suceefully read from in the file\n",iRet);
    printf("Data from the file is   :   \n");

    write(1,Data,iRet);
    close(fd);  //Closing the file

return 0;
}