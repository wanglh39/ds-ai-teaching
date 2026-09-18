#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct HTNode {
    char *key;
    int64_t value;
    struct HTNode *next;
} HTNode;

typedef struct {
    HTNode **buckets;
    size_t capacity;
    size_t size;
    double max_load;
} HashTable;

void ht_init(HashTable *ht, size_t capacity);
void ht_free(HashTable *ht);
bool ht_put(HashTable *ht, const char *key, int64_t value);
bool ht_get(const HashTable *ht, const char *key, int64_t *out);
bool ht_has(const HashTable *ht, const char *key);
bool ht_remove(HashTable *ht, const char *key);
size_t ht_size(const HashTable *ht);
double ht_load_factor(const HashTable *ht);
size_t ht_capacity(const HashTable *ht);

#endif