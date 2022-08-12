#include<stdio.h>
int main(){
	char arr[6]="Hello";
	char brr[]="Hello";

	char crr[] = {'H','e','l','l','o','\0'};

	char *p =arr;
	while(*p != '\0'){
		printf("%c",*p);
		p++;
	}

	for(p=arr; *p!= '\0'; p++){
		printf("%c",*p);
	}

return 0;
}