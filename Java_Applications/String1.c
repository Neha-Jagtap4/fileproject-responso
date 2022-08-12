#include <stdio.h>
#include <stdlib.h>

int main(int argC, char const *argU[]){
	int ino1=0,ino2=0,ians=0;
	printf("Total number number of arugment :%d\n",argC);

	if(argC == 3){ 
		ino1=atoi(argU[1]);
		ino2=atoi(argU[2]);
		ians = ino1+ino2;
	printf("Addition is :%d\n",ians);
	}
	return 0;
}
