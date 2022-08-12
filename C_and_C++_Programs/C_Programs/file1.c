//Neha.txt file create
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
int main(){
    int fd=0;
    char Fname[20];

    printf("Enter file name\n");
    scanf("%s",Fname);
    
    fd=creat(Fname, 0.777);

    if(fd==-1){
        printf("Unable to create the files\n");
    }
    else{
        printf("File sucessfully made");
    }
return 0;
}

/*#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
int main(){
    int fd=0;
    
    fd=creat("Neha.txt", 0.777);

    if(fd==-1){
        printf("Unable to create the files\n");
    }
    else{
        printf("File sucessfully made");
    }
return 0;
}*/