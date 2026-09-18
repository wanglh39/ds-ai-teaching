#include "dlist.h"
#include <stdlib.h>

void dl_init(DList *l) {
    l->head = NULL;
    l->tail = NULL;
    l->size = 0;
}

void dl_free(DList *l) {
    DListNode *cur = l->head;
    while (cur) {
        DListNode *next = cur->next;
        free(cur);
        cur = next;
    }
    l->head = NULL;
    l->tail = NULL;
    l->size = 0;
}

static DListNode *make_node(int key, int value) {
    DListNode *n = malloc(sizeof(DListNode));
    if (!n) return NULL;
    n->key = key;
    n->value = value;
    n->prev = NULL;
    n->next = NULL;
    return n;
}

DListNode *dl_push_front(DList *l, int key, int value) {
    DListNode *n = make_node(key, value);
    if (!n) return NULL;
    if (!l->head) {
        l->head = l->tail = n;
    } else {
        n->next = l->head;
        l->head->prev = n;
        l->head = n;
    }
    l->size++;
    return n;
}

DListNode *dl_push_back(DList *l, int key, int value) {
    DListNode *n = make_node(key, value);
    if (!n) return NULL;
    if (!l->tail) {
        l->head = l->tail = n;
    } else {
        n->prev = l->tail;
        l->tail->next = n;
        l->tail = n;
    }
    l->size++;
    return n;
}

bool dl_pop_front(DList *l, int *out_key, int *out_value) {
    if (!l->head) return false;
    DListNode *n = l->head;
    if (out_key) *out_key = n->key;
    if (out_value) *out_value = n->value;
    l->head = n->next;
    if (l->head) l->head->prev = NULL;
    else l->tail = NULL;
    free(n);
    l->size--;
    return true;
}

bool dl_pop_back(DList *l, int *out_key, int *out_value) {
    if (!l->tail) return false;
    DListNode *n = l->tail;
    if (out_key) *out_key = n->key;
    if (out_value) *out_value = n->value;
    l->tail = n->prev;
    if (l->tail) l->tail->next = NULL;
    else l->head = NULL;
    free(n);
    l->size--;
    return true;
}

void dl_unlink(DList *l, DListNode *node) {
    if (node->prev) node->prev->next = node->next;
    else l->head = node->next;
    if (node->next) node->next->prev = node->prev;
    else l->tail = node->prev;
    node->prev = NULL;
    node->next = NULL;
    l->size--;
}

void dl_move_to_front(DList *l, DListNode *node) {
    if (node == l->head) return;
    dl_unlink(l, node);
    node->next = l->head;
    if (l->head) l->head->prev = node;
    l->head = node;
    if (!l->tail) l->tail = node;
    l->size++;
}

void dl_remove(DList *l, DListNode *node) {
    dl_unlink(l, node);
    free(node);
}

size_t dl_size(const DList *l) { return l->size; }