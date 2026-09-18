#include "skiplist.h"

#include <stdlib.h>

static SkipNode *skipnode_alloc(int level, int key, int value) {
    SkipNode *node = (SkipNode *)malloc(sizeof(SkipNode) + sizeof(SkipNode *) * level);
    node->key = key;
    node->value = value;
    node->level = level;
    for (int i = 0; i < level; i++) {
        node->next[i] = NULL;
    }
    return node;
}

void skiplist_init(SkipList *sl) {
    sl->head = skipnode_alloc(SKIPLIST_MAX_LEVEL, 0, 0);
    sl->level = 1;
    sl->count = 0;
}

void skiplist_free(SkipList *sl) {
    SkipNode *node = sl->head->next[0];
    while (node) {
        SkipNode *next = node->next[0];
        free(node);
        node = next;
    }
    free(sl->head);
    sl->head = NULL;
    sl->level = 0;
    sl->count = 0;
}

int skiplist_random_level(void) {
    int level = 1;
    while ((rand() & 1) && level < SKIPLIST_MAX_LEVEL) {
        level++;
    }
    return level;
}

void skiplist_insert(SkipList *sl, int key, int value) {
    SkipNode *update[SKIPLIST_MAX_LEVEL];
    SkipNode *x = sl->head;
    for (int i = sl->level - 1; i >= 0; i--) {
        while (x->next[i] && x->next[i]->key < key) {
            x = x->next[i];
        }
        update[i] = x;
    }
    int lvl = skiplist_random_level();
    if (lvl > sl->level) {
        for (int i = sl->level; i < lvl; i++) {
            update[i] = sl->head;
        }
        sl->level = lvl;
    }
    SkipNode *node = skipnode_alloc(lvl, key, value);
    for (int i = 0; i < lvl; i++) {
        node->next[i] = update[i]->next[i];
        update[i]->next[i] = node;
    }
    sl->count++;
}

bool skiplist_search(const SkipList *sl, int key, int *out_value) {
    const SkipNode *x = sl->head;
    for (int i = sl->level - 1; i >= 0; i--) {
        while (x->next[i] && x->next[i]->key < key) {
            x = x->next[i];
        }
    }
    x = x->next[0];
    if (x && x->key == key) {
        if (out_value) *out_value = x->value;
        return true;
    }
    return false;
}

bool skiplist_delete(SkipList *sl, int key) {
    SkipNode *update[SKIPLIST_MAX_LEVEL];
    SkipNode *x = sl->head;
    for (int i = sl->level - 1; i >= 0; i--) {
        while (x->next[i] && x->next[i]->key < key) {
            x = x->next[i];
        }
        update[i] = x;
    }
    x = x->next[0];
    if (!x || x->key != key) {
        return false;
    }
    for (int i = 0; i < x->level; i++) {
        if (update[i]->next[i] == x) {
            update[i]->next[i] = x->next[i];
        }
    }
    while (sl->level > 1 && sl->head->next[sl->level - 1] == NULL) {
        sl->level--;
    }
    free(x);
    sl->count--;
    return true;
}

bool skiplist_find_le(const SkipList *sl, int key, int *out_key, int *out_value) {
    const SkipNode *x = sl->head;
    for (int i = sl->level - 1; i >= 0; i--) {
        while (x->next[i] && x->next[i]->key <= key) {
            x = x->next[i];
        }
    }
    if (x == sl->head) {
        return false;
    }
    if (out_key) *out_key = x->key;
    if (out_value) *out_value = x->value;
    return true;
}

bool skiplist_find_ge(const SkipList *sl, int key, int *out_key, int *out_value) {
    const SkipNode *x = sl->head;
    for (int i = sl->level - 1; i >= 0; i--) {
        while (x->next[i] && x->next[i]->key < key) {
            x = x->next[i];
        }
    }
    x = x->next[0];
    if (!x) {
        return false;
    }
    if (out_key) *out_key = x->key;
    if (out_value) *out_value = x->value;
    return true;
}

bool skiplist_find_closest(const SkipList *sl, int key, int *out_key, int *out_value) {
    int le_key, le_val;
    bool has_le = skiplist_find_le(sl, key, &le_key, &le_val);
    int ge_key, ge_val;
    bool has_ge = skiplist_find_ge(sl, key, &ge_key, &ge_val);
    if (!has_le && !has_ge) {
        return false;
    }
    if (!has_le) {
        *out_key = ge_key;
        *out_value = ge_val;
        return true;
    }
    if (!has_ge) {
        *out_key = le_key;
        *out_value = le_val;
        return true;
    }
    if (key - le_key <= ge_key - key) {
        *out_key = le_key;
        *out_value = le_val;
    } else {
        *out_key = ge_key;
        *out_value = ge_val;
    }
    return true;
}

int skiplist_size(const SkipList *sl) {
    return sl->count;
}

int skiplist_max_level(const SkipList *sl) {
    return sl->level;
}