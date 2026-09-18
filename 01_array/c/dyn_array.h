#ifndef DYN_ARRAY_H
#define DYN_ARRAY_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    int *data;
    size_t size;
    size_t capacity;
} DynArray;

void da_init(DynArray *a);
void da_free(DynArray *a);
bool da_append(DynArray *a, int v);
int da_get(const DynArray *a, size_t i);
bool da_set(DynArray *a, size_t i, int v);
bool da_pop(DynArray *a, int *out);
bool da_resize(DynArray *a, size_t new_cap);
void da_clear(DynArray *a);
size_t da_size(const DynArray *a);
size_t da_capacity(const DynArray *a);

#endif