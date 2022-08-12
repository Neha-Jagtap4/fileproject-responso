#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<fcntl.h>
#include<unistd.h>
#include<string.h>
using namespace std;

class MarvellousFile{
    public:
    char Fname[20];
    int fd=0;

    MarvellousFile(char *Name)
    {
        cout<<"Inside constructor";
        strcpy(Fname,Name);
        fd=open(Fname,O_RDWR);

    }

    ~MarvellousFile(){
        cout<<"Inside destructor"<<"\n";
        close(fd);
    }

    void Display(){
        int iRet=0;
        char Buffer[10];
        lseek(fd,0,0);

        cout<<"Data from the file"<<"\n";
        while((iRet=read(fd,Buffer,10))!=0){
            write(1,Buffer,iRet);

        }
    }

    int CountCapital(){
      int iRet=0,iCnt=0,i=0;
      char Buffer[10];
      lseek(fd,0,0);

      while ((iRet=read(fd,Buffer,10))!=0){
          for(i=0;i<iRet;i++){
              if(Buffer[i]>='A' && Buffer[i]<='Z'){
                iCnt++;
              }
           }
        }
    
    
    return iCnt;
    }
};
      
int main(){
    char Name[30];
    int iRet=0;

    cout<<"Enter the file name"<<"\n";
    cin>>Name;

    MarvellousFile mobj(Name);
    mobj.Display();
    iRet=mobj.CountCapital();
    cout<<"Capital lteer count is "<<iRet<<"\n";

return 0;
}