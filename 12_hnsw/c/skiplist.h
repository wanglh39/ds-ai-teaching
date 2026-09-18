#ifndef SKIPLIST_H
#define SKIPLIST_H

#include <stdbool.h>

#define SKIPLIST_MAX_LEVEL 16

typedef struct SkipNode {
    int key;
    int value;
    int level;
    struct SkipNode *next[];
} SkipNode;

typedef struct {
    SkipNode *head;
    int level;
    int count;
} SkipList;

void skiplist_init(SkipList *sl);
void skiplist_free(SkipList *sl);
int skiplist_random_level(void);
void skiplist_insert(SkipList *sl, int key, int value);
bool skiplist_search(const SkipList *sl, int key, int *out_value);
bool skiplist_delete(SkipList *sl, int key);
bool skiplist_find_le(const SkipList *sl, int key, int *out_key, int *out_value);
bool skiplist_find_ge(const SkipList *sl, int key, int *out_key, int *out_value);
bool skiplist_find_closest(const SkipList *sl, int key, int *out_key, int *out_value);
int skiplist_size(const SkipList *sl);
int skiplist_max_level(const SkipList *sl);

#endif