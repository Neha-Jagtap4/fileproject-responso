//Neha.txt file
//It  shows the 21 bytes in the file(properties ,madhe bhagyach)
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
#include<fcntl.h>

int main(){
    int fd=0,iRet=0,iCnt=0,iSum=0;
    char Fname[10];
    char Buffer[10];

    printf("Enter file name\n");  //Neha.txt
    scanf("%s",Fname);

    fd=open(Fname,O_RDWR);
    if(fd==-1){
        printf("Unable to open file");
        return -1;
    }
               //kay,kasha,kiti  buffer he Wheat madhe kahade sarkh aah aadhi first 10,nantr10,nantr 10 pppppn last madhe ab ch tr te2 parayant firn (10,2) condition satifies madhe jail na toh
    while((iRet=read(fd,Buffer,10))!= 0){   //He imp aah ((())!=0)
        iSum=iSum+iRet;
        
    }
    printf("Total number of a is: %d\n",iSum);
    
    close(fd);
    
    
return 0;
}