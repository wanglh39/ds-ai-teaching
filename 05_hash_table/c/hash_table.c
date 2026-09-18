#include "hash_table.h"
#include <stdlib.h>
#include <string.h>

static size_t ht_hash(const char *key, size_t capacity) {
    size_t h = 0;
    const size_t A = 31;
    while (*key) {
        h = h * A + (unsigned char)(*key);
        key++;
    }
    return h % capacity;
}

static HTNode *ht_node_new(const char *key, int64_t value) {
    HTNode *node = (HTNode *)malloc(sizeof(HTNode));
    if (!node) return NULL;
    node->key = (char *)malloc(strlen(key) + 1);
    if (!node->key) {
        free(node);
        return NULL;
    }
    strcpy(node->key, key);
    node->value = value;
    node->next = NULL;
    return node;
}

static void ht_node_free(HTNode *node) {
    if (node) {
        free(node->key);
        free(node);
    }
}

void ht_init(HashTable *ht, size_t capacity) {
    if (capacity == 0) capacity = 16;
    ht->capacity = capacity;
    ht->size = 0;
    ht->max_load = 0.75;
    ht->buckets = (HTNode **)calloc(capacity, sizeof(HTNode *));
}

void ht_free(HashTable *ht) {
    for (size_t i = 0; i < ht->capacity; i++) {
        HTNode *node = ht->buckets[i];
        while (node) {
            HTNode *next = node->next;
            ht_node_free(node);
            node = next;
        }
    }
    free(ht->buckets);
    ht->buckets = NULL;
    ht->capacity = 0;
    ht->size = 0;
}

static bool ht_rehash(HashTable *ht, size_t new_capacity) {
    HTNode **new_buckets = (HTNode **)calloc(new_capacity, sizeof(HTNode *));
    if (!new_buckets) return false;
    for (size_t i = 0; i < ht->capacity; i++) {
        HTNode *node = ht->buckets[i];
        while (node) {
            HTNode *next = node->next;
            size_t idx = ht_hash(node->key, new_capacity);
            node->next = new_buckets[idx];
            new_buckets[idx] = node;
            node = next;
        }
    }
    free(ht->buckets);
    ht->buckets = new_buckets;
    ht->capacity = new_capacity;
    return true;
}

bool ht_put(HashTable *ht, const char *key, int64_t value) {
    size_t idx = ht_hash(key, ht->capacity);
    HTNode *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            node->value = value;
            return true;
        }
        node = node->next;
    }
    HTNode *new_node = ht_node_new(key, value);
    if (!new_node) return false;
    new_node->next = ht->buckets[idx];
    ht->buckets[idx] = new_node;
    ht->size++;
    if ((double)ht->size / (double)ht->capacity > ht->max_load) {
        ht_rehash(ht, ht->capacity * 2);
    }
    return true;
}

bool ht_get(const HashTable *ht, const char *key, int64_t *out) {
    size_t idx = ht_hash(key, ht->capacity);
    HTNode *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            if (out) *out = node->value;
            return true;
        }
        node = node->next;
    }
    return false;
}

bool ht_has(const HashTable *ht, const char *key) {
    return ht_get(ht, key, NULL);
}

bool ht_remove(HashTable *ht, const char *key) {
    size_t idx = ht_hash(key, ht->capacity);
    HTNode *prev = NULL;
    HTNode *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            if (prev) prev->next = node->next;
            else ht->buckets[idx] = node->next;
            ht_node_free(node);
            ht->size--;
            return true;
        }
        prev = node;
        node = node->next;
    }
    return false;
}

size_t ht_size(const HashTable *ht) {
    return ht->size;
}

double ht_load_factor(const HashTable *ht) {
    return ht->capacity > 0 ? (double)ht->size / (double)ht->capacity : 0.0;
}

size_t ht_capacity(const HashTable *ht) {
    return ht->capacity;
}