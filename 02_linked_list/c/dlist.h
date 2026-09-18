#ifndef DLIST_H
#define DLIST_H

#include <stddef.h>
#include <stdbool.h>

typedef struct DListNode {
    int key;
    int value;
    struct DListNode *prev;
    struct DListNode *next;
} DListNode;

typedef struct {
    DListNode *head;
    DListNode *tail;
    size_t size;
} DList;

void dl_init(DList *l);
void dl_free(DList *l);

DListNode *dl_push_front(DList *l, int key, int value);
DListNode *dl_push_back(DList *l, int key, int value);

bool dl_pop_front(DList *l, int *out_key, int *out_value);
bool dl_pop_back(DList *l, int *out_key, int *out_value);

void dl_move_to_front(DList *l, DListNode *node);
void dl_remove(DList *l, DListNode *node);

size_t dl_size(const DList *l);

#endif