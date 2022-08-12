//Accept the whole file and then display the file.
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
#include<fcntl.h>
//#define O_RDWR
int main(){
    int fdsrc=0,fddest=0,iRet=0;
    char Fname1[20];
    char Fname2[20];
    char Buffer[1024]; //Marvellous  hellos //it is like mug //1024 byte concept kran kr for company

    printf("Enter file name\n");
    scanf("%s",Fname1);
    
    fdsrc=open(Fname1,O_RDONLY);

    if(fdsrc==-1){
        printf("Unable to OPEN the files\n");
        return -1;
    }
    else{
        printf("File sucessfully Opened with fd :%d\n",fdsrc);
    }

    printf("Enter the fddest file name\n");
    scanf("%s",Fname2);
    fddest=creat(Fname2,0.777);

    while(iRet = read(fdsrc,Buffer,sizeof(Buffer))!=0){
        write(fdsrc,Buffer,iRet);
    }
    
    close(fdsrc);
    close(fddest);  //Closing the file

return 0;
}