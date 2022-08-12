#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<io.h>
#include<fcntl.h>
int CountCapital(char Fname[]){
    int iRet=0,fd=0,iCnt=0,i=0;
    char Buffer[10];
    fd=open(Fname,O_RDWR);
    if(fd==-1){
        printf("Unable to open file");
        return -1;
    }
               //kay,kasha,kiti  buffer he Wheat madhe kahade sarkh aah aadhi first 10,nantr10,nantr 10 pppppn last madhe ab ch tr te2 parayant firn (10,2) condition satifies madhe jail na toh
    while((iRet=read(fd,Buffer,10))!= 0){   //He imp aah ((())!=0)
        for(i=0; i<iRet; i++){ //10 parayant firn
            if(Buffer[i]=='a'){  //Wheat madhe Kahade.. aali tr count hoil
                iCnt++;
            }
        }
    }
    printf("Total number of a is: %d\n",iCnt);
    
    close(fd);
}
int main(){
    int iRet=0;
    char Fname[10];
   // char Buffer[10];

    printf("Enter file name\n");  //Neha.txt
    scanf("%s",Fname);

    /*fd=open(Fname,O_RDWR);
    if(fd==-1){
        printf("Unable to open file");
        return -1;
    }
               //kay,kasha,kiti  buffer he Wheat madhe kahade sarkh aah aadhi first 10,nantr10,nantr 10 pppppn last madhe ab ch tr te2 parayant firn (10,2) condition satifies madhe jail na toh
    while((iRet=read(fd,Buffer,10))!= 0){   //He imp aah ((())!=0)
        for(i=0; i<iRet; i++){ //10 parayant firn
            if(Buffer[i]=='a'){  //Wheat madhe Kahade.. aali tr count hoil
                iCnt++;
            }
        }
    }
    printf("Total number of a is: %d\n",iCnt);
    
    close(fd);*/
    CountCapital(Fname);
    //printfNumber of capital are %d\n",iRet);
return 0;
}

//file chi size 47 aasel tr file chi size 47 ani 0
//file chi size 44  asel tr file chi size 44 ani 0