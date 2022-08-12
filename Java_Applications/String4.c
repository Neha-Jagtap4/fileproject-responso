#include <stdio.h>
#include<string.h>

int main(){
	char arr[20] = "Hello";
	char brr[20];
	char crr[20]="World";

	int iret=0;
	//printf("Size of arr is %d\n" ,sizeof(arr));
	//printf("Length of arr is :%d\n",strlen(arr));

	strcpy(brr,arr); //Strcpy is use to copy the string
	printf("Content of brr is :%s\n",brr);

	iret = strcmp(arr,brr);
	if(iret == 0){
		printf("String is equal");
	}

	else{
		printf("String is not equal");
	}

	strcat(arr,crr);
	printf("String after concentration : %s\n",arr);
return 0;
}