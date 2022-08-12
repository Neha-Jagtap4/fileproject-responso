#include <stdio.h>

#include <stdlib.h>

int main(int argC,char *argU[]){
	int i=0;
	printf("Total number of command %d\n",argC );

	for(int i=0;i<argC; i++){
		printf("%s\n",argU[i]);
	}
	return 0;
}