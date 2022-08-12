#include<iostream>
#include<stdio.h>

using namespace std;

typedef struct node{
    int data;
    struct node *next;
}NODE, *PNODE, **PPNODE;

void InsertFirst(PPNODE Head,int no){
    PNODE newn =new NODE;
    newn->data = no;
    newn->next = NULL;

    if(*Head == NULL){
        *Head = newn;
    }

    else{
        newn->next = *Head;
        *Head = newn;
    }
}

void Display(PNODE Head){
    //PNODE temp = Head;
    if(Head != NULL){
        
        cout<<"|"<<Head->data<<"\n";
        Head= Head->next;
        Display(Head);
    }
}

int main(){
    PNODE first = NULL;

    InsertFirst(&first,101);
    InsertFirst(&first,11);
    InsertFirst(&first,21);

    Display(first);

    return 0;
}